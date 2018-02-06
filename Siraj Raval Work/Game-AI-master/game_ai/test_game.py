from environment import Environment
from train import Trainer
from dqn import DQN
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
env = Environment(args)
agent = DQN(env, args)

Trainer(agent).run()

env.gym.monitor.stat(args.out, force=True)
agent.play()
env.gym.monitor.close()