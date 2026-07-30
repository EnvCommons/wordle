import asyncio
import textarena as ta
import re
import nltk
from typing import List, Tuple
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


# TextArena's Wordle constructs an EnglishDictionary on every ta.make(), and that
# constructor calls nltk.download("words") unconditionally. Even when the corpus is
# already present, nltk.download does a live network round-trip to the nltk server
# to check for a newer version. The env-server runs on a single asyncio event loop,
# so that blocking network call stalls *every* in-flight session on the pod at once;
# under RL concurrency /ping, /prompt and /delete all blow past the proxy's 10s
# timeout and surface as a storm of 502s. Ensure the datasets TextArena needs are
# present once at import, then make nltk.download a no-op so per-session env
# construction never touches the network again.
#
# NB: the no-op neuters *all* nltk.download calls, so every dataset TextArena's
# Wordle relies on must be pre-ensured here: the "words" corpus and the
# "averaged_perceptron_tagger_eng" tagger (used by pos_tag when filtering the
# word list).
_NLTK_REQUIREMENTS = (
    ("corpora/words", "words"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)


def _ensure_nltk_data() -> None:
    for find_path, package in _NLTK_REQUIREMENTS:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(package)


_ensure_nltk_data()
nltk.download = lambda *args, **kwargs: True


class TaskSpec(BaseModel):
    id: str
    env_id: str
    seed: int
    variant: str = ""


class GuessParams(BaseModel, extra="forbid"):
    word: str


class WordleEnvironment(Environment):
    GAME_NAME = "Wordle"
    BASE_ENV_ID = "Wordle-v0"
    VARIANTS = [
        "Wordle-v0",
        "Wordle-v0-hardcore",
        "Wordle-v0-long",
        "Wordle-v0-long-hardcore",
    ]
    NUM_TASKS_PER_VARIANT = 50

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.ta_env = ta.make(env_id=self.config.env_id)
        self.game_done = False
        self.turn_count = 0

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        tasks = []
        for variant_id in cls.VARIANTS:
            for seed_idx in range(cls.NUM_TASKS_PER_VARIANT):
                seed = seed_idx if split == "train" else seed_idx + 10000
                tasks.append({
                    "id": f"{variant_id}_seed{seed}",
                    "env_id": variant_id,
                    "seed": seed,
                    "variant": variant_id,
                })
        return tasks

    def _format_observation(self, observation) -> str:
        if isinstance(observation, str):
            match = None
            for m in re.finditer(r'^\[(?!GAME\])[^\]]+\].*$', observation, re.MULTILINE):
                match = m
            if match:
                return observation[match.end():].lstrip('\n')
            return observation
        if isinstance(observation, list):
            if not observation:
                return ""
            last = observation[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return str(last[1])
            return str(last)
        return str(observation)

    async def get_prompt(self) -> List[TextBlock]:
        # reset()/step()/close() are synchronous TextArena calls; run them off the
        # event loop so they can't stall other sessions sharing this pod's loop.
        await asyncio.to_thread(self.ta_env.reset, num_players=1, seed=self.config.seed)
        _, observation = self.ta_env.get_observation()
        obs_text = self._format_observation(observation)
        prompt = (
            f"You are playing Wordle.\n\n"
            f"{obs_text}\n\n"
            f"Use the guess_word tool to submit your guesses. Each guess must be a valid English word "
            f"of the correct length. After each guess you receive feedback:\n"
            f"  - G (green): correct letter in the correct position\n"
            f"  - Y (yellow): letter exists in the word but wrong position\n"
            f"  - X (wrong): letter is not in the word\n"
        )
        return [TextBlock(text=prompt)]

    @tool
    async def guess_word(self, params: GuessParams) -> ToolOutput:
        """Submit a word guess. The word must be a valid English word of the correct length."""
        if self.game_done:
            return ToolOutput(
                blocks=[TextBlock(text="Game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True,
            )

        action = f"[{params.word}]"
        done, info = await asyncio.to_thread(self.ta_env.step, action=action)
        self.turn_count += 1

        if done:
            self.game_done = True
            rewards, game_info = await asyncio.to_thread(self.ta_env.close)
            # TextArena's Wordle already returns a reward in [0, 1] (1.0 on a
            # correct guess, otherwise a continuous completion fraction), so
            # pass it through unchanged rather than remapping an already-
            # normalised value through (raw + 1) / 2, which compressed every
            # outcome into [0.5, 1.0] and paid a total failure 0.5.
            reward = rewards.get(0, 0.0) if isinstance(rewards, dict) else float(rewards)
            reason = ""
            if isinstance(game_info, dict) and 0 in game_info:
                reason = game_info[0].get("reason", "")
            summary = f"Game Over! Reward: {reward:.2f}"
            if reason:
                summary += f"\n{reason}"
            return ToolOutput(
                blocks=[TextBlock(text=summary)],
                metadata={"turn": self.turn_count, "reward": reward},
                reward=reward,
                finished=True,
            )

        _, observation = self.ta_env.get_observation()
        obs_text = self._format_observation(observation)
        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False,
        )
