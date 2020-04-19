import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import numpy as np
import random
import math
import time
from collections import deque
from test_game import TestGame
import matplotlib
from matplotlib import pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from pacman_game import PacmanGame
#Be sure to have installed all the libraries stated above, and tensorflow!



class Q_learning:
    #Creating the models
    def __init__(self, gamma_ = 0.95, epsilon_ = 1, epsilon_decay_ = 0.995, epsilon_min_ = 0.01, lr_ = 0.001, tau_ = 0.01, layers_ = 5, nodes_b_ = 300, nodes_h1_ = 200, nodes_h2_ = 150, nodes_h3_ = 100, activation_ = "tanh", loss_ = "mean_squared_error"):
        self.memory = deque(maxlen = 10000) #Make an empty deque to store information, which is a list-like datatype that can append data faster than a normal list. Maxlen is 800, as it will have enough batches to learn from by then.
        #Hyperparameters, to be tweaked by GA when initialising a NN with its unique 'i' hyperparameters.
        self.gamma = gamma_ #Discount factor. Every reward gets multiplied by gamma after each step, lowering the importance of initial reward.
        self.epsilon = epsilon_ #'Random' factor. The higher this value, the more not random choices are made (more reliant on the NN's weights and biases).
        self.epsilon_decay = epsilon_decay_ #Factor with which epsilon lowers after every step. This way, the NN is 'trusted' more and more as it has more training data.
        self.epsilon_min = epsilon_min_ #Lowest value of epsilon, to ensure the NN still takes a random step sometimes, as it might accidently find a better policy --> helps against falling into local optima.
        self.lr =  lr_ #How big is the step we take in teh direction of current reward.
        self.tau = tau_ #Parameter to ensure that the step taken is not too big --> How much are we taking the last policy update into account?
        self.NN_output_size = 4  #Number of inputs for the game (left, right, turn around)
        self.NN_input_shape = (404,) #Size of the state inputted by the game
        #NOTE: two models are created here, as DeepMind has proven this leads to quicker convergence in complex environments.
        #Where self.model keeps changing its 'ideal policy' with each iteration (where is my next food pallet?), target_model keeps the 'end goal' in sight (highest eventual score).
        self.model = self.create_model(layers = layers_, nodes_b = nodes_b_, nodes_h1 = nodes_h1_, nodes_h2 = nodes_h2_, nodes_h3 = nodes_h3_, activation__ = activation_, loss__ = loss_)
        self.target_model = self.create_model(layers = layers_, nodes_b = nodes_b_, nodes_h1 = nodes_h1_, nodes_h2 = nodes_h2_, nodes_h3 = nodes_h3_, activation__ = activation_, loss__ = loss_)
        self.min_observe = 32
        #self.model.save_weights("weights5L.h5") Save weights
        #print(self.model.get_weights())


    def create_model(self, layers = 3, nodes_b = 4, nodes_h1 = 200, nodes_h2 = 150, nodes_h3 = 20, activation__ = "tanh", loss__ = "mean_squared_error"):
        model = Sequential() #Using the Built-in 'Sequential' architecture --> layer for layer in sequential order from input to output.
        #Now adding layers:
        model.add(Dense(nodes_b, input_shape=self.NN_input_shape)) #Dense forward progegates. Furthermore, the parameters are (output_layer, initial_input, activation). activation_ & loss_ are used to prevent possible self-referencing errors
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        if layers >= 3:
            model.add(Dense(nodes_h1)) #After the initial layer, the amount of inputs does not need to be specified.
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dropout(0.2))
        if layers >= 4:
            model.add(Dense(nodes_h2)) #After the initial layer, the amount of inputs does not need to be specified.
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dropout(0.2))
        if layers >= 5:
            model.add(Dense(nodes_h3)) #After the initial layer, the amount of inputs does not need to be specified.
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dropout(0.2))
        model.add(Dense(self.NN_output_size, activation = "softmax")) #Changed it so we don't want a probability anymore, but instead make it calculate the expected reward
        model.compile(loss=loss__, optimizer = SGD(lr = self.lr))
        #Like Stochastic Gradient Descent (SGD), Adam is a optimization algorithm to optimize the policy of this NN. Unlike SGD, Adam is able to optimize the NN based on iterative based training data (--> no need for historical, labelled data).
        #The loss can be chosen from the Keras set by the GA, but is chosen as mean_squared_error (most straightforward) for now.
        # print("model summary:", model.summary())
        # print("model inputs:", model.inputs)
        # print("model output:", model.outputs)
        return(model)

    #Saving information of a run:
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
         #For each iteration, add to the memory what action has been taken in which environment (state), what reward has been given and what the new state is.
    
    #Training the Neural Network:
    def replay(self, batch_size = 16):
        if len(self.memory) < self.min_observe: #sets a minimum amount of observations to learn from
            return
        else:
            samples = random.sample(self.memory, batch_size) #Using the random library, we pick a random batch of size batch-size of entries from the memory.
            for sample in samples: #For the amount of samples:
                state, action, reward, new_state, done = sample
                
                state = state.reshape((1,404))
                # print("new state: ", new_state.shape)
                new_state = new_state.reshape((1,404))

                target = self.target_model.predict(state)
                if done: #If the Learning process is finished / converged, is the future reward is equal to the next step.
                    target[0][action] = reward #Direct reward that is given with these actions
                else:
                    Q_future = max(self.target_model.predict(new_state)[0]) #Cummulative future rewards
                    # print("Q Future: ", Q_future)
                    target[0][action] = reward + Q_future * self.gamma #The target of the self.model is to get the best current reward combined with all expected future rewards, multiplied by gamma.
                # print("State: ", state)
                # print("Target: ", target)
                self.model.train_on_batch(state, target, sample_weight=None, class_weight=None, reset_metrics=True)
                self.model.fit(state, target, epochs=250, verbose = 0) #Trains the number for a given amount of epochs #verbose = 1 shows a progress bar of how far you are with regards to the total amount of epochs 

            
    def target_train(self): #Training the target model less frequently, to make sure its goal is more consistent over time.
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)): #Replace all weights with target weights.
            target_weights[i] = weights[i] #Maybe we can remove this loop and just say target_weights = weights.copy()
        self.target_model.set_weights(target_weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay #Update the epsilon value.
        self.epsilon = max(self.epsilon_min, self.epsilon) #If epsilon is lower than the threshold value, take the threshold value.
        if len(self.memory) < self.min_observe: #We don't want to take policy action if there is nothing learnt yet
            return np.random.randint(0,4)
        if np.random.random() < self.epsilon: #There is a epsilon probability that we will take a random action, instead of the action that seems to be best.
            return np.random.randint(0,4) #Take a random integer, representing either doign nothing, turning left, right, or turn around.

        # print("In act state shape: ", state.shape)
        state = state.reshape((1,404))
        # print("New state shape: ", state.shape)
        # print(self.model.predict(state))
        print(self.model.predict(state))
        return np.argmax(self.model.predict(state)[0])

    def save(self, file_name):
        self.model.save_weights(file_name + ".h5")
#changed epislon_r to 0.1 for testing

def main(saving = False, file_name = "Final_weights", gamma_r = 0.95, epsilon_r = 1, epsilon_decay_r = 0.99, epsilon_min_r = 0.01, lr_r = 0.05, tau_r = 0.05, layers_r = 5, nodes_b_r = 1000, nodes_h1_r = 1000, nodes_h2_r = 1000, nodes_h3_r = 500, activation_r = "relu", loss_r = "mean_squared_error", batch_size_r = 32, trials_r = 100, trial_len_r = 800): #Integrate all hyperparameters into relevant functions.
    game = PacmanGame() #initialising PacmanGame
    trials = trials_r
    trial_len = trial_len_r

    network = Q_learning(gamma_ = gamma_r, epsilon_ = epsilon_r, epsilon_decay_ = epsilon_decay_r, epsilon_min_ = epsilon_min_r, lr_ = lr_r, tau_ = tau_r, layers_ = layers_r, nodes_b_ = nodes_b_r, nodes_h1_ = nodes_h1_r, nodes_h2_ = nodes_h2_r, nodes_h3_ = nodes_h3_r, activation_ = activation_r, loss_ = loss_r)
    rewards_his = []
    for trial in range(trials):
        cur_state, dummy_1, dummy_2 = game.update(3) #Start the game and do nothing, to initialise and get the first environment
        rewards = 0 #The cummulative score for a specific game
        print("Trial:", trial)
        for step in range(trial_len):
            # print("step: ",step, end='\r')
            action = network.act(cur_state) #Create an action to take
            #print("State: ", cur_state, "Action: ", action)
            new_state, reward, done = game.update(action) #TO BE LOOKED INTO, correct outputs have to be given.
            # print("state: ", new_state.shape)
            # print("Total reward", rewards)
            rewards += reward
            network.remember(cur_state, action, reward, new_state, done) #Remember all these parameters, to learn later.
            network.replay(batch_size = batch_size_r) #Replay with a batch from the memory, updating the model every sample of the batch_size.
            network.target_train() #Train the target function once.

            cur_state = new_state
            if done:
                final_score = rewards
                rewards_his.append(final_score)
                print("Total steps: ", step)
                print("final score: ", final_score)
                game.restart()
                break
    if saving:
        network.save(file_name)
    return(rewards_his, final_score) #if final_score doesn't work, return rewards_his[-1]

###The Genetic Algorithm
#For this project, the gen_length of the 'organisms' is equal to the amount of hyperparameters that have been indicated with 'hyperparameter_i' in the code, being 

def F(genome): #The fitness function inputs a certain amount of hyperparameters into the NN_Basic, and outputs its accuracy minus a training time penalty.

    #Pairs genome to actual hyperparameters. Given the initialisation between 0 and 100 for each genome entry, the code below makes it likely for the GA to pick 'sweet-spot values' of hyperparameters between 0 and 1. --> Faster convergence.
    #A quick summary of what genome entry rules over which hyperparameter:
    #genome[0] = gamma
    #genome[1] = epsilon_decay
    #genome[2] = learning rate (lr)
    #genome[3] = tau
    #genome[4] = amount of layers

    if genome[0] >= 1: #Value of gamma
        gamma_i = genome[1]/100
    elif genome[0] <= 0:
        gamma_i = 0.00000001 #to avoid /0 error, or get negative logs.
    else:
        gamma_i = genome[0]

    if genome[1] >= 1: #Epsilon decay value.
        epsilon_decay_i = genome[1]/100
    elif genome[1] <= 0:
        epsilon_decay_i = 0.00000001 #to avoid /0 error, or get negative logs.
    else:
        epsilon_decay_i = genome[1]
    
    if genome[2] >= 1: #learning rate
        lr_i = genome[2]/100
    elif genome[2] <= 0:
       lr_i = 0.00000001 #to avoid /0 error, or get negative logs.
    else:
        lr_i = genome[2]
    
    if genome[3] >= 1: #tau value, usually around 0.001
        tau_i = genome[3]/1000
    elif genome[3] <= 0:
        tau_i = 0.00000001 #to avoid /0 error, or get negative logs.
    else:
        tau_i = genome[3]
    #Number of total layers, which has a hard-coded max of 5 layers. Higher numbers will result in 5 layers to be created.
    layers_i = 2 #in- and output layer.
    if genome[4] > 25:
        layers_i += 1
    if genome[4] > 50:
        layers_i += 1
    if genome[4] > 75:
        layers_i += 1
    #Now that all hyperparameters are entered, it is time to actually start the fitness function.
    t_0 = time.perf_counter()
    rewards_his, final_score = main(gamma_r = gamma_i, epsilon_decay_r = epsilon_decay_i, lr_r = lr_i, tau_r = tau_i, layers_r = layers_i)
    fitness_1 = final_score
    t_f = time.perf_counter()
    Delta_t = (t_f - t_0) * 0.001 #Penalty to equalise points with training time's influence (it should be heavily penalised if calculations take too long, since it would slow down the training of all of the NNs significantly if they all take a lot of time).
    fitness = fitness_1 - Delta_t
    return(fitness)

def cross(A, B):
    # Randomly pick half of the indeces of A & B (dependant on the fact of Choice_n is 0 or 1, respectively),
    # Ensuring that the values of gene C_1 is either a value of A_1 or B_1.
    Choice = np.random.choice(np.append(np.zeros(math.floor(np.size(A)/2)), np.ones(math.ceil(np.size(A)/2))), np.size(A), replace=False) #Make a vector with the length of the genome, with an equal amount of randomized 0's and 1's.
    C = np.array([])
    for i in range(np.size(Choice)):
        if Choice[i] == 0: #If the entry in the Choice vector (C_i) is 0, pick the entry from A (A_i).
            C = np.append(C, A[i])
        elif Choice[i] == 1: #If the entry in the Choice vector (C_i) is 1, pick the entry from B (B_i).
            C = np.append(C, B[i])
    return(C)

def mutate(C, mut_range = 0.5, mut_chance = 0.2): #ARBITRARY VALUES!
    #mut_range is the range of the amplitude of the mutation is going from -mut_range to mut_range, to ensure both larger & smaller, positive & negatives values for the new mutation.
    #mut_chance is the chance of the mutation actually occuring.
    for i in range(np.size(C)):
        if np.random.uniform(0,1) <= mut_chance: #For a 5% chance, mutate (perform addition) C_i with either 0.95 or 1.05 (slightly increasing or decreasing the value of an allele)
            mut_factor = np.random.choice(np.arange(1-mut_range, 1+mut_range, 0.01).tolist()) #Increases randomness of mutation.
            C[i] = C[i] * mut_factor
    return(C)

def cross_and_mutate(pop, pop_size):
    # pop describes the complete population:
    size_population = pop_size[0]
    N = int(size_population/4)
    gen_length = pop_size[1]  #number of genomes. For this NN, gen_length = number of hyperparameters
    offspring = np.zeros(pop_size)
    for k in range(N): #top 25% gets into the new generation
        offspring[k] = pop[k]
    k = N
    while k < size_population:
        if (k < 2 * N): #The second quartile of the old population gets used to cross with one another for offspring in the new generation.
            offspring[k] = cross(pop[k],pop[k+1])
        if (k >= 2 * N) and (k < 3*N): #The third quartile of the old population gets used to cross with a random other instance in the pop for offspring in the new generation.
            r_1 = np.random.randint(0, size_population)
            offspring[k] = cross(pop[k], pop[r_1])
        if (k >= 3 * N): # For the last quartile of entries into the new population, two random entries in the old population get crossed for offspring in the new population.
            r_1 = np.random.randint(0, size_population)
            r_2 = np.random.randint(0, size_population)
            offspring[k] = cross(pop[r_1], pop[r_2])
        k = k + 1
    for i in range(offspring.shape[0]): #Once the new generation is created, let every individual go through a 'mutation check'.
        mutate(offspring[i])
    return offspring

def run(N = 10, gen_length = 5, num_generations = 10, Fitness_function = F):
    size_population = 4 * N #the actual amount of individuals
    pop_size = (size_population, gen_length) #The entire population, described in terms of its population size and the amount of alleles per individual.
    new_population = np.ndarray((pop_size)) #Generate gen 0, with random values for each allele between 0 and 100. <--- ARBITRARY VALUES, FEEL FREE TO TWEAK!
    for individual in range(size_population):
        for gene in range(gen_length):
            new_population[individual, gene] = np.random.randint(0, 101) #int instead of flt to highly decrease calculation time
    #print(new_population)

    fitness = np.zeros(size_population) #initialise the fitness function as an array in which the fitness of all individuals can be described.
    fitness_his = [] #This variable will be used to plot the fitness over iterations.
    for generation in range(num_generations):
        #print("#Generation:", generation)
        for individual in range(size_population):
            #print("#Individual: ", individual+1,"/", size_population)
            fitness[individual] = Fitness_function(new_population[individual]) #The fitness is determined by the values of the genes of that individual
        Ranking = np.argsort(fitness)[::-1] #Ranking individuals from highest to lowest fitness value.
        fitness = fitness[Ranking] #Sort the fitness according to the ranks
        fitness_his = np.append(fitness_his, fitness[0]) #Add the highest fitness value as entry for this generation to the history.
        new_population = new_population[Ranking] #Sort the new population according to the ranks
        offspring_crossover = cross_and_mutate(new_population, pop_size) #Cross and mutate for the new generation with the ordered old generation.
        new_population = offspring_crossover #Initialise the new generation as the generation to be ranked, etc.
        print("Generation", generation, "has a highest fitness of", fitness[0], "with genome", new_population[0])
    for individual in range(size_population): #The entire loop is repeated one last time for the final generation.
            print("#Individual: ", individual+1,"/", size_population)
            fitness[individual] = Fitness_function(new_population[individual])
    Ranking = np.argsort(fitness)[::-1]
    fitness = fitness[Ranking]
    new_population = new_population[Ranking]
    #print("Fitness:",fitness[0],"New_Population",new_population[0],"Fitness_his",fitness_his)
    return(fitness[0], new_population[0], fitness_his)

'''Here, one can test if a particular set of hyperparameters result in positive outcomes of the training. The weights will be saved in an accompanying document if saving = True'''
rewards_his, final_score = main()
plt.plot(rewards_his)
plt.xlabel("Number of Trials")
plt.ylabel("Score")
plt.title("DQN [characteristic parameters]")
plt.show()

'''Lastly, the GA can be turned on to find the best hyperparameters using the following line:'''
#best_fitness, best_genome, fitness_his = run()