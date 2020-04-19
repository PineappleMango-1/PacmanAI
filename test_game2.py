import numpy as np
from numpy import random
class Test2:
    def __init__(self):
        self.state = self.get_state()
        self.i = 0
        self.done = False
    def get_state(self):
        choice = np.random.randint(1023)
        bin = self.decimalToBinary(choice)
        dec = self.binaryToDecimal(bin)
        return np.asarray(bin)
    def decimalToBinary(self, num):
        dec = num
        bin = [0,0,0,0,0,0,0,0,0,0]
        for i in range(1000):
            if dec > 1:
                bin[-1-i] = dec%2
                dec = dec//2
            elif dec == 1:
                bin[-1-i] = 1
                break
        return(bin)
    def binaryToDecimal(self, binary): 
        dec = 0
        for i in range(len(binary)):
            if binary[-i-1] == 1:
                dec += 2**i
        return dec
    '''def convert_input(self, input):
        return 2**input'''
    def update(self, input):
        reward = -(abs(input - np.sqrt(self.binaryToDecimal(self.state)))**2)
        self.state = self.get_state()
        if self.i == 100:
            self.done = True
        self.i += 1
        return self.state, reward, self.done
    def restart(self):
        self.done = False
        self.i = 0
    


