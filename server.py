import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openreward.environments import Server
from env import WordleEnvironment

if __name__ == "__main__":
    server = Server([WordleEnvironment])
    server.run()
