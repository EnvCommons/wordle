import textarena as ta
from typing import List, Tuple
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


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

    async def get_prompt(self) -> List[TextBlock]:
        self.ta_env.reset(num_players=1, seed=self.config.seed)
        _, observation = self.ta_env.get_observation()
        obs_text = observation if isinstance(observation, str) else str(observation)
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

    def _map_reward(self, raw_reward: float) -> float:
        return max(0.0, min(1.0, (raw_reward + 1.0) / 2.0))

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
        done, info = self.ta_env.step(action=action)
        self.turn_count += 1

        if done:
            self.game_done = True
            rewards, game_info = self.ta_env.close()
            raw = rewards.get(0, 0.0) if isinstance(rewards, dict) else float(rewards)
            reward = self._map_reward(raw)
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
        obs_text = observation if isinstance(observation, str) else str(observation)
        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False,
        )
