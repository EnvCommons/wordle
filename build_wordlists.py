"""Precompute TextArena Wordle word lists at image-build time.

TextArena's WordleEnv rebuilds its word list on every construction by running nltk
pos_tag over the entire dictionary (the full ~200k-word "en" list for the hardcore
variants) to keep only nouns of the right length. That is seconds of CPU per
session, and the env-server runs it synchronously on its single asyncio event loop
inside /create — which starves every co-tenant session on the pod and produces a
storm of proxy 502s under an RL run's concurrency.

The filtering is deterministic in (hardcore, word_length), so we run it once here,
at build time, and bake the result into the image as wordlists.json. env.py loads
that file and replaces WordleEnv._load_word_list with a cache lookup.
"""

import json

import nltk
import textarena as ta
from textarena.envs.Wordle.env import WordleEnv

# Must match WordleEnvironment.VARIANTS in env.py.
VARIANTS = [
    "Wordle-v0",
    "Wordle-v0-hardcore",
    "Wordle-v0-long",
    "Wordle-v0-long-hardcore",
]

OUTPUT_PATH = "wordlists.json"


def main() -> None:
    for package in ("words", "averaged_perceptron_tagger_eng"):
        nltk.download(package)

    captured: dict[str, list[str]] = {}
    original = WordleEnv._load_word_list

    def capture(self, hardcore: bool = False) -> None:
        # Run the real (expensive) filtering, then record it keyed by the only
        # inputs it depends on: hardcore and word_length.
        original(self, hardcore=hardcore)
        captured[f"{int(bool(hardcore))}:{self.word_length}"] = self.word_list

    WordleEnv._load_word_list = capture
    try:
        for env_id in VARIANTS:
            ta.make(env_id=env_id)
    finally:
        WordleEnv._load_word_list = original

    if not captured:
        raise SystemExit("No word lists were captured; refusing to write an empty cache.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(captured, f)

    summary = ", ".join(f"{key}={len(words)} words" for key, words in sorted(captured.items()))
    print(f"Wrote {len(captured)} word lists to {OUTPUT_PATH}: {summary}")


if __name__ == "__main__":
    main()
