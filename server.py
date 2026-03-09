from openreward.environments import Server
from env import WordleEnvironment

if __name__ == "__main__":
    server = Server([WordleEnvironment])
    server.run()
