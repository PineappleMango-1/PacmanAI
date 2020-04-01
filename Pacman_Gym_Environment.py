import gym
from pacman_game import PacmanGame

class PacmanEnv(gym.Env):

    def __init__(self):
        print("Init")

    def step(self):
        print("Foo")

    def reset(self):
        print("Foo")

    def render(self):
        print("Foo")



env = gym.make('CartPole-v0')
env.reset()
env.render()
