import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from pacman_game import PacmanGame

game = PacmanGame()
NN_output_size = 3 #Number of inputs for the game (left, right, turn around).

class Q_learning:
    #Creating the models
    def __init__(self, game, gamma_i = 0.9, epsilon_i = 1, epsilon_decay_i = 0.995, epsilon_min_i = 0.01, lr_i = 1, tau_i = 0.1):
        self.env = game
        self.memory = deque(maxlen = 2000) #Make an empty deque to store information, which is a list-like datatype that can append data faster than a normal list.
        #Hyperparameters, to be tweaked by GA.
        self.gamma = gamma_i #Discount factor. Every reward gets multiplied by gamma after each step, lowering the importance of initial reward.
        self.epsilon = epsilon_i #'Random' factor. The higher this value, the more not random choices are made (more reliant on the NN's weights and biases).
        self.epsilon_decay = epsilon_decay_i #Factor with which epsilon lowers after every step. This way, the NN is 'trusted' more and more as it has more training data.
        self.epsilon_min = epsilon_min_i #Lowest value of epsilon, to ensure the NN still takes a random step sometimes, as it might accidently find a better policy --> helps against falling into local optima.
        self.lr =  lr_i #How big is the step we take in teh direction of current reward.
        self.tau = tau_i #Parameter to ensure that the step taken is not too big --> How much are we taking the last policy update into account?
        
        #NOTE: two models are created here, as DeepMind has proven this leads to quicker convergence in complex environments.
        #Where self.model keeps changing its 'ideal policy' with each iteration (where is my next food pallet?), target_model keeps the 'end goal' in sight (highest eventual score).
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self, layers_i = 3 nodes_b_i = 50, nodes_h1_i = 40, nodes_h2_i = 30, nodes_h3_i = 20, activation_i = "relu", loss_i = "mean_squared_error"):
        model = Sequential() #Using the Built-in 'Sequential' architecture --> layer for layer in sequential order from input to output.
        #Now adding layers:
        model.add(Dense(nodes_b_i, input_shape=(X.shape), activation = activation_i)) #Dense forward progegates. Furthermore, the parameters are (output_layer, initial_input, activation)
        if layers_i >= 3:
            model.add(Dense(nodes_h1_i, activation = activation_i)) #After the initial layer, the amount of inputs does not need to be specified.
        if layers_i >= 4:
            model.add(Dense(nodes_h2_i, activation = activation_i))
        if layers_i >= 5:
            model.add(Dense(nodes_h3_i, activation = activation_i))
        model.add(Dense(NN_output_size)))
        model.compile(loss = loss_i, optimizer = Adam(lr = self.lr))
        #Like Stochastic Gradient Descent (SGD), Adam is a optimization algorithm to optimize the policy of this NN. Unlike SGD, Adam is able to optimize the NN based on iterative based training data (--> no need for historical, labelled data).
        #The loss can be chosen from the Keras set by the GA, but is chosen as mean_squared_error (most straightforward) for now.
        return(model)
    #NOTE TO SELF: Do not forget to change activation_i in Fitness_function of GA (Incorporate string usage).

    #Saving information of a run:
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done]) #For each iteration, add to the memory what action has been taken in which environment (state), what reward has been given and what the new state is.
    
    #Training the Neural Network:
    def replay(self, batch_size_i = 32):
        batch_size = batch_size_i
        if len(self.memory) < batch_size: #We can not use a batch size bigger than the amount of entries in the memory. So, first we create a memory of this length without learning.
            return

        samples = random.sample(self.memory, batch_size) #Using the random library, we pick a random batch of size batch-size of entries from the memory.
        for sample in samples: #For the amount of samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done: #If the Learning process is finished / converged, is the future reward is equal to the next step.
                target[0][action] = reward #Direct reward that is given with these actions
            else:
                Q_future = max(self.target_model.predict(new_state)[0]) #Cummulative future rewards
                target[0][action] = reward + Q_future * self.gamma #The target of the self.model is to get the best current reward combined with all expected future rewards, multiplied by gamma.
            self.model.fit(state, target, epochs=5, verbose=1) #Trains the number for a given amount of epochs #verbose = 1 shows a progress bar of how far you are with regards to the total amount of epochs 
    
    def target_train(self): #Training the target model less frequently, to make sure its goal is more consistent over time.
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)): #Replace all weights with target weights.
            target_weights[i] = weights[i] #Maybe we can remove this loop and just say target_weights = weights.copy()
        self.target_model.set_weights(target_weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay #Update the epsilon value.
        self.epsilon = max(self.epsilon_min, self.epsilon) #If epsilon is lower than the threshold value, take the threshold value.
        if np.random.random() < self.epsilon: #There is a epsilon probability that we will take a random action, instead of the action that seems to be best.
            return np.random.randint(0,4) #Take a random integer, representing either doign nothing, turning left, right, or turn around.
        return np.argmax(self.model.predict(state) [0])

def main(): #Integrate all hyperparameters into relevant functions.
    gamma = gamma_i
    epsilon = epsilon_min_i

    trails = 100
    trail_len = 500

    model = Q_learning
    #steps = []
    for trail in range(trails):
        cur_state = game.get_gameOutput() #Grabs the environment from the last frame and one-hot encodes it into input for the NN, with 12 possible classes for each entry.
        for step in range(trial_len):
            action = model.act(cur_state) #Create an action to take
            new_state, reward, done = game.update(action) #TO BE LOOKED INTO, correct outputs have to be given.

            model.remember(cur_state, action, reward, new_state, done) #Remember all these parameters, to learn later.
            model.replay() #Replay with a batch from the memory.
            model.target_train() #Train the target function once.

            cur_state = new_state
            if done:
                break
main() #Run the programme.
    