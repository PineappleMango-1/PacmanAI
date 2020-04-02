import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pacman_game import PacmanGame

class PacmanEnv(gym.Env):

    def __init__(self):
        print("Init")

    def step(self, action):
        print("Foo")

    def reset(self):
        print("Foo")

    def render(self, mode='human'):
        print("Foo")



env = gym.make('CartPole-v0')
env.reset()

