import asyncio
import os
from env import WordleEnvironment


def get_secrets():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    secrets = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    secrets[key.strip().lower()] = val.strip()
    return secrets


async def smoke_test():
    print("=== Smoke Testing: Wordle ===\n")

    splits = WordleEnvironment.list_splits()
    print(f"Splits: {splits}")

    tasks = WordleEnvironment.list_tasks(split="test")
    print(f"Test tasks: {len(tasks)}")
    print(f"First task: {tasks[0]}")

    secrets = get_secrets()
    env = WordleEnvironment(task_spec=tasks[0], secrets=secrets)

    prompt = await env.get_prompt()
    print(f"\nPrompt ({len(prompt[0].text)} chars):")
    print(prompt[0].text[:500])

    from env import GuessParams
    result = await env.guess_word(GuessParams(word="SLATE"))
    print(f"\nGuess 'SLATE' result:")
    print(f"  Reward: {result.reward}")
    print(f"  Finished: {result.finished}")
    print(f"  Output: {result.blocks[0].text[:300]}")

    print("\n=== Smoke test PASSED ===")


if __name__ == "__main__":
    asyncio.run(smoke_test())
