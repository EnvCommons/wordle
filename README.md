# Wordle

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Wordle)

## Description

**Wordle** is an environment for evaluating agents on the word-guessing game Wordle. The agent must deduce a hidden target word by submitting guesses and interpreting positional feedback (correct, misplaced, or absent letters). This environment wraps the Wordle implementation from [TextArena](https://github.com/LeonGuertwordle/TextArena), a framework for text-based game environments.

## Capabilities

- Logical deduction from partial information
- Vocabulary and word frequency knowledge
- Constraint satisfaction (incorporating feedback across multiple guesses)
- Information-theoretic reasoning (maximizing information gain per guess)

## Compute Requirements

Wordle does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (200 tasks) and test (200 tasks). Each split contains 50 tasks across each of 4 variants:

- **Wordle-v0**: Standard 5-letter Wordle with 6 guesses
- **Wordle-v0-hardcore**: Hardcore mode requiring guesses to incorporate all previous feedback
- **Wordle-v0-long**: Extended version with longer words
- **Wordle-v0-long-hardcore**: Extended version with hardcore constraints

Each task is seeded for reproducibility. The target word is randomly selected from a dictionary at reset time.

## Reward Structure

This is a sparse reward environment. TextArena's Wordle reward is already in `[0.0, 1.0]` and is passed through unchanged: guessing the target word scores `1.0`, and every other terminal outcome (running out of guesses or an invalid guess) scores a continuous completion fraction based on how close the final guess was. Intermediate guesses score `0.0`.

We do not use LLM graders for this environment; reward is determined programmatically by exact match against the target word.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given a single tool:

- `guess_word(word)`: Submit a word guess. The word must be a valid English word of the correct length.

## Time Horizon

Wordle is a multi-turn environment. The agent submits one guess per turn and receives feedback.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; Wordle works out of the box without any secrets or API keys.

## Safety

Agents in Wordle interact only with a word-guessing game and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
