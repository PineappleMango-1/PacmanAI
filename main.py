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
    def __init__(self, gamma_ = 0.9, epsilon_ = 1, epsilon_decay_ = 0.995, epsilon_min_ = 0.01, lr_ = 1, tau_ = 0.1, layers_ = 3, nodes_b_ = 50, nodes_h1_ = 40, nodes_h2_ = 30, nodes_h3_ = 20, activation_ = "relu", loss_ = "mean_squared_error"):
        self.memory = deque(maxlen = 2000) #Make an empty deque to store information, which is a list-like datatype that can append data faster than a normal list. Maxlen as a failsafe.
        #Hyperparameters, to be tweaked by GA when initialising a NN with its unique 'i' hyperparameters.
        self.gamma = gamma_ #Discount factor. Every reward gets multiplied by gamma after each step, lowering the importance of initial reward.
        self.epsilon = epsilon_ #'Random' factor. The higher this value, the more not random choices are made (more reliant on the NN's weights and biases).
        self.epsilon_decay = epsilon_decay_ #Factor with which epsilon lowers after every step. This way, the NN is 'trusted' more and more as it has more training data.
        self.epsilon_min = epsilon_min_ #Lowest value of epsilon, to ensure the NN still takes a random step sometimes, as it might accidently find a better policy --> helps against falling into local optima.
        self.lr =  lr_ #How big is the step we take in teh direction of current reward.
        self.tau = tau_ #Parameter to ensure that the step taken is not too big --> How much are we taking the last policy update into account?
        
        #NOTE: two models are created here, as DeepMind has proven this leads to quicker convergence in complex environments.
        #Where self.model keeps changing its 'ideal policy' with each iteration (where is my next food pallet?), target_model keeps the 'end goal' in sight (highest eventual score).
        self.model = self.create_model(layers = layers_, nodes_b = nodes_b_, nodes_h1 = nodes_h1_, nodes_h2 = nodes_h2_, nodes_h3 = nodes_h3_, activation = activation_, loss = loss_)
        self.target_model = self.create_model(layers = layers_, nodes_b = nodes_b_, nodes_h1 = nodes_h1_, nodes_h2 = nodes_h2_, nodes_h3 = nodes_h3_, activation = activation_, loss = loss_)

    def create_model(self, layers = 3, nodes_b = 50, nodes_h1 = 40, nodes_h2 = 30, nodes_h3 = 20, activation_ = "relu", loss_ = "mean_squared_error"):
        model = Sequential() #Using the Built-in 'Sequential' architecture --> layer for layer in sequential order from input to output.
        #Now adding layers:
        model.add(Dense(nodes_b, input_shape=(X.shape), activation = activation_)) #Dense forward progegates. Furthermore, the parameters are (output_layer, initial_input, activation). activation_ & loss_ are used to prevent possible self-referencing errors
        if layers >= 3:
            model.add(Dense(nodes_h1, activation = activation_)) #After the initial layer, the amount of inputs does not need to be specified.
        if layers >= 4:
            model.add(Dense(nodes_h2, activation = activation_))
        if layers >= 5:
            model.add(Dense(nodes_h3, activation = activation_))
        model.add(Dense(NN_output_size, activation = "softmax"))) #Since we want the NN to output a probability, softmax is used in the final layer.
        model.compile(loss = loss_, optimizer = Adam(lr = self.lr))
        #Like Stochastic Gradient Descent (SGD), Adam is a optimization algorithm to optimize the policy of this NN. Unlike SGD, Adam is able to optimize the NN based on iterative based training data (--> no need for historical, labelled data).
        #The loss can be chosen from the Keras set by the GA, but is chosen as mean_squared_error (most straightforward) for now.
        return(model)
    #NOTE TO SELF: Do not forget to change activation_i in Fitness_function of GA (Incorporate string usage).

    #Saving information of a run:
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done]) #For each iteration, add to the memory what action has been taken in which environment (state), what reward has been given and what the new state is.
    
    #Training the Neural Network:
    def replay(self, batch_size = batch_size_):
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

def main(gamma_r = 0.9, epsilon_r = 1, epsilon_decay_r = 0.995, epsilon_min_r = 0.01, lr_r = 1, tau_r = 0.1, layers_r = 3, nodes_b_r = 50, nodes_h1_r = 40, nodes_h2_r = 30, nodes_h3_r = 20, activation_r = "relu", loss_r = "mean_squared_error", batch_size_r = 32): #Integrate all hyperparameters into relevant functions.
    gamma = gamma_r
    epsilon = epsilon_r

    trails = 100
    trail_len = 500

    network = Q_learning(gamma_ = gamma_r, epsilon_ = epsilon_r, epsilon_decay_ = epsilon_decay_r, epsilon_min_ = epsilon_min_r, lr_ = lr_r, tau_ = tau_r, layers_ = layers_r, nodes_b_ = nodes_b_r, nodes_h1_ = nodes_h1_r, nodes_h2_ = nodes_h2_r, nodes_h3_ = nodes_h3_r, activation_ = activation_r, loss_ = loss_r)
    #steps = []
    for trail in range(trails):
        cur_state = game.get_gameOutput() #Grabs the environment from the last frame and one-hot encodes it into input for the NN, with 12 possible classes for each entry.
        for step in range(trial_len):
            action = network.act(cur_state) #Create an action to take
            new_state, reward, done = game.update(action) #TO BE LOOKED INTO, correct outputs have to be given.

            network.remember(cur_state, action, reward, new_state, done) #Remember all these parameters, to learn later.
            network.replay(batch_size = batch_size_r) #Replay with a batch from the memory.
            network.target_train() #Train the target function once.

            cur_state = new_state
            if done:
                break #Return final reward score? This way we can use it in the GA to get the fitness function rather quickly.


###The Genetic Algorithm
#For this project, the gen_length of the 'organisms' is equal to the amount of hyperparameters that have been indicated with 'hyperparameter_i' in the code, being 

def F(genome): #The fitness function inputs a certain amount of hyperparameters into the NN_Basic, and outputs its accuracy
    #n_epochs_i = int(math.ceil(genome[0])) ##Needs to be a whole number number that is not 0 --> integer & ceiling. Currently not used, due to Elena's advise.
    
    #Pairs genome to actual hyperparameters. Given the initialisation between 0 and 100 for each genome entry, the code below makes it likely for the GA to pick 'sweet-spot values' of hyperparameters between 0 and 1. --> Faster convergence.
    #A quick summary of what genome entry rules over which hyperparameter:
    #genome[0] = gamma
    #genome[1] = epsilon
    #genome[2] = epsilon_decay
    #genome[3] = epsilon_min
    #genome[4] = learning rate (lr)
    #genome[5] = tau
    #genome[6] = batch_size
    #genome[7] = activation (function)
    #genome[8] = loss (function)
    #genome[9] = amount of layers
    #genome[10] = amount of nodes in hidden second layer
    #genome[11] = amount of nodes in hidden third layer
    #genome[12] = amount of nodes in hidden fourth layer
    #genome[13] = amount of nodes in hidden first layer (after input)
    if genome[0] >= 1: #Value of gamma
        gamma_i = genome[1]/100
    elif genome[0] <= 0:
        gamma_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        gamma_i = genome[0]

    if genome[1] >= 1: #Epsilon value.
        epsilon_i = genome[1]/100
    elif genome[1] <= 0:
        epsilon_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        epsilon_i = genome[1]

    if genome[2] >= 1: #epsilon decay value
        epsilon_decay_i = genome[2]/100
    elif genome[2] <= 0:
        epsilon_decay_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        epsilon_decay_i = genome[2]
    
    if genome[3] >= 1: #epsilon_min value
        epsilon_min_i = genome[3]/100
    elif genome[3] <= 0:
        epsilon_min_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        epsilon_min_i = genome[3]
    
    if genome[4] >= 1: #learning rate
        lr_i = genome[4]/100
    elif genome[4] <= 0:
       lr_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        lr_i = genome[4]
    
    if genome[5] >= 1: #tau value
        tau_i = genome[5]/100
    elif genome[5] <= 0:
        tau_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        tau_i = genome[5]

    if genome[6] <= 0: #Batch Size
        batch_size_i = 1
    else:
        batch_size_i = int(math.ceil(genome[6])) #Needs to be a whole number --> integer & ceil (so it doesn't become 0)

    #In Keros, there are 9 activation functions available for hidden layers which do not neat tuning, being described below:
    activation_list = ["relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear", "softmax", "softplus", "softsign"]
    for i in range(activation_list.size): #Grabs an activation based on the size of the genome.
        if i == 0 and genome[7] < 100/(activation_list.size):
            activation_i = activation_list[i]
        elif 100/(activation_list.size/i)< genome[7] <= 100/(activation_list.size/i+1):
            activation_i = activation_list[i]
        else:
            activation_i = "linear" #Safeguard if genome[7] > 100, picked a function that is generally a bad activation function (does nothing).

    #In Keros, there are 12 loss functions that do not require specific inputs (for example, only categorical targets)
    loss_list = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "logcosh", "huber_loss", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
    for i in range(loss_list.size):
        if i == 0 and genome[8] < 100/(loss_list.size):
            loss_i = loss_list[i]
        elif 100/(loss_list.size/i)< genome[8] <= 100/(loss_list.size/i+1):
            loss_i = loss_list[i]
        else:
            loss_i = "mean_squared_error" #Safeguard if genome[8] > 100. I do not have a good understanding of loss functions, so I took the first one I saw (--> arbitrary).


    #Number of total layers, which has a hard-coded max of 5 layers. Higher numbers will result in 5 layers to be created.
    layers_i = 2 #in- and output layer.
    if genome[9] > 25
        layers_i += 1
    if genome[9] > 50:
        layers_i += 1
    if genome[9] > 75:
        layers_i += 1

    #For nodes per layer, we might want to find a more efficient way (Maybe if layers_i == 3 for example, only load nodes_h1_i)
    if genome[10] > 200: #Nodes in first hidden layer
        nodes_h1_i = 200
    elif genome[10] <= 1:
        nodes_h1_i = 1
    else:
        nodes_h1_i = int(math.ceil(genome[10])) #Ceil is arbitrarely chosen here

    if genome[11] > 200: #Nodes in second hidden layer
        nodes_h2_i = 200
    elif genome[11] <= 1:
        nodes_h2_i = 1
    else:
        nodes_h2_i = int(math.ceil(genome[11])) #Ceil is arbitrarely chosen here

    if genome[12] > 200: #Nodes in third hidden layer
        nodes_h3_i = 200
    elif genome[12] <= 1:
        nodes_h3_i = 1
    else:
        nodes_h3_i = int(math.ceil(genome[12])) #Ceil is arbitrarely chosen here

    if genome[13] > 200: #Nodes in the first layer, actually being the first true 'hidden' layer already.
        nodes_b_i = 200
    elif genome[13] <= 1:
        nodes_b_i = 1
    else:
        nodes_b_i = int(match.ceil(genome[13]))

    #Now that all hyperparameters are entered, it is time to actually start the fitness function.
    t_0 = time.perf_counter()
    #reward = main(gamma_r = gamma_i, epsilon_r = epsilon_i, epsilon_decay_r = epsilon_decay_i, epsilon_min_r = epsilon_min_i, lr_r = lr_i, tau_r = tau_i, layers_r = layers_i, nodes_b_r = nodes_b_i, nodes_h1_r = nodes_h1_i, nodes_h2_r = nodes_h2_i, nodes_h3_r = nodes_h3_i, activation_r = activation_i, loss_r = loss_i, batch_size_r = batch_size_i)
    #fitness_1 = 1/reward.
    t_f = time.perf_counter()
    Delta_t = (t_f - t_0) * 0.01 #Penalty to equalise points with training time's influence (it should be heavily penalised if calculations take too long, since it would slow down the training of all of the NNs significantly if they all take a lot of time).
    fitness = fitness_1 + Delta_t
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

def run(N = 14, gen_length = 14, num_generations = 10, Fitness_function = F):
    size_population = 4 * N #the actual amount of individuals
    pop_size = (size_population, gen_length) #The entire population, described in terms of its population size and the amount of alleles per individual.
    new_population = np.ndarray((pop_size)) #Generate gen 0, with random values for each allele between 0 and 100. <--- ARBITRARY VALUES, FEEL FREE TO TWEAK!
    for individual in range(size_population):
        for gene in range(gen_length):
            new_population[individual, gene] = np.random.randint(0, 101) #int instead of flt to highly decrease calculation time
    print(new_population)

    fitness = np.zeros(size_population) #initialise the fitness function as an array in which the fitness of all individuals can be described.
    fitness_his = [] #This variable will be used to plot the fitness over iterations.
    for generation in range(num_generations):
        for individual in range(size_population):
            fitness[individual] = Fitness_function(new_population[individual]) #The fitness is determined by the values of the genes of that individual
        Ranking = np.argsort(fitness)[::1] #Ranking individuals from lowest to highest fitness value.
        fitness = fitness[Ranking] #Sort the fitness according to the ranks
        fitness_his = np.append(fitness_his, fitness[0]) #Add the highest fitness value as entry for this generation to the history.
        new_population = new_population[Ranking] #Sort the new population according to the ranks
        offspring_crossover = cross_and_mutate(new_population, pop_size) #Cross and mutate for the new generation with the ordered old generation.
        new_population = offspring_crossover #Initialise the new generation as the generation to be ranked, etc.
        print("Generation", generation, "has a highest fitness of", fitness[0], "with genome", new_population[0])
    for individual in range(size_population): #The entire loop is repeated one last time for the final generation.
            fitness[individual] = Fitness_function(new_population[individual])
    Ranking = np.argsort(fitness)
    fitness = fitness[Ranking]
    return(fitness[0], new_population[0], fitness_his)