import numpy as np
from numpy import random
class TestGame:
    def __init__(self):
        self.state = [1,0,0,0]
        self.i = 0
        self.done = False
    def get_rand(self):
        self.choice = [0,0,0,0]
        self.choice[np.random.randint(4)] = 1
        self.choice = np.asarray(self.choice)
        return self.choice
    def get_input(self, input):
        self.OH = [0,0,0,0]
        self.OH[input] = 1
        self.OH = np.asarray(self.OH)
        return self.OH
    def update(self, NN_input):
        self.same = False
        self.inputs = self.get_input(NN_input)
        if self.done:
            self.done = False
        if self.i == 50:
            self.done = True
        for i in range(4):
            if self.inputs[i] == self.state[i]:
                self.same = True
            else:
                self.same = False
                self.reward = -0.1
                break
        if self.same:
            self.reward = 1
        self.state = self.get_rand()
        self.i+=1
        return self.state, self.reward, self.done
    def restart(self):
        self.state = [1,0,0,0]
        self.i = 0
        self.done = False


#game = TestGame()
#for i in range(110):
    #guess = np.random.randint(4)
    #state, rewarad, done = game.go(guess)
    #print("Guess: ", guess, "State: ", state, "Reward: ", rewarad, "Done: ", done)


