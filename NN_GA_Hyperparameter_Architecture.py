import numpy as np
import struct
import math
import matplotlib
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from os.path import join
# One-hot encoding


def one_hot(Y, n_class):
    # accept a (1xm) 'label' vector, number of classes   <--- e.g. Y = [1, 3, 5, 8] if x_0 -> x_4 corresponds to these classes. (y_0 = 1, y_1 = 3, etc.)
    # and return a nxm 'one-hot' matrix                  <--- n_class the amount of different classes.
    Y = np.ravel(Y).astype(int)                 #In later data sets, the entries are arrays of floats instead of integers ([0.0] vs. 0). To let the code work, this needs to be changed
    O = np.zeros((n_class, Y.size))        #Initiate a matrix with n_class rows (amount of classes) and an amount of column equal to the amount of examples.
    for i in range(Y.size):                #For every data point with a label
        O[Y[i],i] = 1                      #Turn the i,Y[i] value in the matrix into a '1'.
    return O

#Reverse one-hot encoding
def inv_one_hot(O):
    length = np.shape(O)[1]

    Y = np.zeros((1,length))

    for i in range(length):
        j = np.argmax(O[:,i])
        Y[0,i] = j

    return Y

#Normalizing data
def normalize(X):
    mean = np.mean(X,axis=1,keepdims=True)
    std =  np.std(X,axis=1,keepdims=True)
    N_X = (X-mean)/(std)

    return N_X

#Accuracy of the NN
def model_accuracy(H,Y):
    # Y has to be one-hot matrix
    # H can be either one hot or softmax output.
    n = np.shape(H)[1]
    err = 0
    O = inv_one_hot(H)
    L = inv_one_hot(Y)
    for i in range(n):
        if O[0,i]!=L[0,i]:
            err += 1
    accuracy = (1 - err/n)

    return accuracy

#Accessing the test database
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def flatten_imgs(X):
    n_img = np.shape(X)[0]
    h = np.shape(X)[1]
    w = np.shape(X)[2]
    size_arr = h*w
    return np.reshape(X,(n_img,size_arr))

''' CHANGE PATH IN THE BELOW LINE ACCORDING TO YOUR DIRECTORY STRUCTURE'''
data_folder = "data"
X = read_idx(join(data_folder, 'train-images-idx3-ubyte'))
X = flatten_imgs(X)
X = normalize(X)               # Try without this
X = np.transpose(X)

''' CHANGE PATH IN THE BELOW LINE ACCORDING TO YOUR DIRECTORY STRUCTURE'''
Y = read_idx(join(data_folder, 'train-labels-idx1-ubyte'))
Y = np.expand_dims(Y, axis=1)
Y = np.transpose(Y)
Y = one_hot(Y,10)

''' CHANGE PATH IN THE BELOW LINE ACCORDING TO YOUR DIRECTORY STRUCTURE'''
X_test = read_idx(join(data_folder, 't10k-images.idx3-ubyte'))
X_test = flatten_imgs(X_test)
X_test = normalize(X_test)     # Try without this
X_test = np.transpose(X_test)

''' CHANGE PATH IN THE BELOW LINE ACCORDING TO YOUR DIRECTORY STRUCTURE'''
Y_test = read_idx(join(data_folder, 't10k-labels.idx1-ubyte'))
Y_test = np.expand_dims(Y_test, axis=1)
Y_test = np.transpose(Y_test)
Y_test = one_hot(Y_test,10)

#All Activation Classes
class sigmoid:
    def activate(self,Z):
        A = 1/(1+np.exp(-Z))
        return A

    def diff(self, Z):
        dA_dZ = np.multiply(self.activate(Z),(1-self.activate(Z)))
        return dA_dZ

class relu:
    def activate(self,Z):
        A = np.maximum(0,Z)
        return A

    def diff(self,Z):
        dA_dZ = 1*(Z>0)
        return dA_dZ

class tanh:
    def activate(self,Z):
        A = np.tanh(Z)
        return A

    def diff(self,Z):
        dA_dZ = 1 - (np.multiply(self.activate(Z),self.activate(Z)))
        return dA_dZ

class softmax:
    def activate(self,Z):
        e_Z = np.exp(Z- np.max(Z,axis=0))
        A = e_Z / e_Z.sum(axis=0)
        return A


    def diff(self,Z):
        sftmx = self.activate(Z)
        a = np.einsum('ij,jk->ijk',np.eye(sftmx.shape[0]),sftmx)
        b = np.einsum('ij,kj->ikj',sftmx,sftmx)
        dH_dZ = a - b
        return dH_dZ

#All loss functions
class CE_loss:
    def get_loss(self,H,Y):
        L = np.sum(np.dot(-Y.T,np.log(H)))/Y.shape[1]
        return L

    def diff(self,H,Y):
        n = Y.shape[0]
        dL_dZ = 1/n*(H-Y)
        return dL_dZ

#Initialise theta function
def init_theta(n1,n2,activation):
    #n1 = number of nodes in prev layer (input)
    #n2 = number of nodes in next layer (output)
    #activation is the class of activation (sigmoid, relu, etc.)
    if activation in [sigmoid,softmax]:
        M = np.random.randn(n2,n1)*np.sqrt(2./n1)
    elif activation in [relu] :
        M = np.random.randn(n2,n1)*np.sqrt(1./n1)
    elif activation == tanh:
        M = np.random.randn(n2,n1)*np.sqrt(1./(n1+n2))
    else:
        M = np.random.randn(n2,n1)
    return M




#All gradient descrent functions

#Stogastic gradient descent (using mini batches, works well for shallow networks):
#CHANGE IF WE USE DEEP NN!
def SGD(batch_size,X,Y,model,lr=0.001): #Takes random examples of the training set --> faster gradient descent, accuracy is slightly lower.
    m = np.shape(X)[1]
    for i in range(0,m,batch_size):
        X_batch = X[:,i:i+batch_size]
        Y_batch = Y[:,i:i+batch_size]

        #call model's f_pass() on X_batch
        model.f_pass(X_batch)
        #call model's back_prop() for X_batch,Y_batch and batch_size
        model.back_prop(X_batch, Y_batch, batch_size)
        #call model's optim() for lr
        model.optim(lr)
    return model.loss

#Training + Plotting said training function
def train(model, X, Y, X_test, Y_test, metric, n_epochs=100, batch_size=4, lr=0.01, lr_decay=1):
    data_size = X.shape[1]
    for e in range(n_epochs):
        #Shuffle dataset: helps remove possible serial-relations between data, and reduces 'memorisation'
        #If reinforcement learning is used, this could be changed to experience replay.
        np.random.seed(138)
        shuffle_index = np.random.permutation(data_size)
        X, Y = X[:,shuffle_index], Y[:,shuffle_index]

        #Gradient descent function
        loss = SGD(batch_size,X,Y,model,lr)

        #decay helps decrease the size of steps over time, improving stability and convergence
        lr = lr*lr_decay

        #train accuracy
        H = model.f_pass(X)
        tr_acc = metric(H,Y)

        #test accuracy
        H = model.f_pass(X_test)
        acc = metric(H,Y_test)

        #plot train accuracy and test accuracy vs epochs
        plt.plot(e,tr_acc, 'bo')
        plt.plot(e,acc,'ro')
        clear_output()
        print(f"epoch:{e+1}/{n_epochs} | Loss:{loss:.4f} | \
            Train Accuracy: {tr_acc:.4f} | Test_Accuracy:{acc:.4f}")

    #plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.show()

#Accuracy function: To be used for fitness later on, does not plot all of the trainings
def acc_level(model, X, Y, X_test, Y_test, metric, n_epochs=100, batch_size=4, lr=0.01, lr_decay=1):
    data_size = X.shape[1]
    for e in range(n_epochs):
        #Shuffle dataset: helps remove possible serial-relations between data, and reduces 'memorisation'
        #If reinforcement learning is used, this could be changed to experience replay.
        np.random.seed(138)
        shuffle_index = np.random.permutation(data_size)
        X, Y = X[:,shuffle_index], Y[:,shuffle_index]

        #Gradient descent function
        loss = SGD(batch_size,X,Y,model,lr)

        #decay helps decrease the size of steps over time, improving stability and convergence
        lr = lr*lr_decay

        #train accuracy
        H = model.f_pass(X)
        tr_acc = metric(H,Y)

        #test accuracy
        H = model.f_pass(X_test)
        acc = metric(H,Y_test)
        return(acc)

#Plot data function
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    if axes == None:
        axes = plt.gca()
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);

#Initial hyperparameters
n_epochs = 200
lr = 0.01
lr_decay = 0.99
batch_size = 16
lossfn = CE_loss
activation = relu
layers = 3
nodes_b = 150
nodes_h1 = 50

#More hyperparameters are: layers, lossfn, activation & nodes per layer, number of epochs.


#Here, we can define different architectures
class NN_Basic:
    def __init__(self, X_size, Y_size, lossfn = CE_loss, activation = relu, layers = 3, nodes_b = 150, nodes_h1 = 50, nodes_h2 = 25, nodes_h3 = 10): #<--- nodes_x added for more layers if wanted
        # Build the network. Recommended depth: 2-5 layers. Number of nodes: below 200 preferably.
        # Final activation should be softmax, since each datapoint belongs to only one class
        self.L_b = layer(X_size, nodes_b, activation)
        if layers == 2: #Hard coded because I am not a good coder...
            self.L_f = layer(nodes_b, Y_size, softmax)

        elif layers == 3:
            self.L_h1 = layer(nodes_b, nodes_h1, activation)
            self.L_f = layer(nodes_h1, Y_size, softmax)

        elif layers == 4:
            self.L_h1 = layer(nodes_b, nodes_h1, activation)
            self.L_h2 = layer(nodes_h1, nodes_h2, activation)
            self.L_f = layer(nodes_h1, Y_size, softmax)

        elif layers == 5:
            self.L_h1 = layer(nodes_b, nodes_h1, activation)
            self.L_h2 = layer(nodes_h1, nodes_h2, activation)
            self.L_h3 = layer(nodes_h2, nodes_h3, activation)
            self.L_f = layer(nodes_h1, Y_size, softmax)
        self.lossfn = lossfn()

    def f_pass(self, X):
        A_b = self.L_b.forward(X)
        if layers == 2:
            A_f = self.L_f.forward(A_b)

        elif layers == 3:
            A_h1 = self.L_h1.forward(A_b)
            A_f = self.L_f.forward(A_h1)

        elif layers == 4:
            A_h1 = self.L_h1.forward(A_b)
            A_h2 = self.L_h2.forward(A_h1)
            A_f = self.L_f.forward(A_h2)

        elif layers == 5:
            A_h1 = self.L_h1.forward(A_b)
            A_h2 = self.L_h2.forward(A_h1)
            A_h3 = self.L_h3.forward(A_h2)
            A_f = self.L_f.forward(A_h3)

        self.H = A_f
        return self.H

    def back_prop(self,X,Y, batch_size):
        m = batch_size

        #use the model's lossfn's get_loss() to find the loss of the model
        self.loss = self.lossfn.get_loss(self.H, Y)

        #use the model's lossfn's diff() to find dL_dZ
        dL_dZ = self.lossfn.diff(self.H, Y)

        if layers == 2:
            self.L_f.out_grad(dL_dZ, self.L_b.A, m)
            self.L_b.grad(self.L_f.dZ, self.L_f.W, X, m)

        if layers == 3:
            self.L_f.out_grad(dL_dZ, self.L_h1.A, m)
            self.L_h1.grad(self.L_f.dZ, self.L_f.W, self.L_b.A, m)

        if layers == 4:
            self.L_f.out_grad(dL_dZ, self.L_h2.A, m)
            self.L_h2.grad(self.L_f.dZ, self.L_f.W, self.L_h1.A, m)
            self.L_h1.grad(self.L_h2.dZ, self.L_h2.W, self.L_b.A, m)

        if layers == 5:
            self.L_f.out_grad(dL_dZ, self.L_h3.A, m)
            self.L_h3.grad(self.L_f.dZ, self.L_f.W, self.L_h2.A, m)
            self.L_h2.grad(self.L_h3.dZ, self.L_h3.W, self.L_h1.A, m)
            self.L_h1.grad(self.L_h2.dZ, self.L_h2.W, self.L_b.A, m)

        if layers >= 3:
            self.L_b.grad(self.L_h1.dZ, self.L_h1.W, X, m)

    def optim(self, lr):
        #call each layer's step() to update their respective W and B
        self.L_b.step(lr)
        if layers >= 3:
            self.L_h1.step(lr)
        if layers >= 4:
            self.L_h2.step(lr)
        if layers >= 5:
            self.L_h3.step(lr)
        self.L_f.step(lr)

#Layer class: Defining a layer's forward pass, learning rate, gradient and gradient for the last layer.
class layer:
    def __init__(self, n_prev, n_next, activation):

        #Each layer object has  W, B and activation()
        self.W = init_theta(n_prev, n_next, activation)
        self.B = init_theta(1, n_next, activation)
        self.activation = activation()

    def forward(self, A_prev):
        self.Z =  np.dot(self.W, A_prev) + self.B
        self.A = self.activation.activate(self.Z)
        return self.A

    def grad(self, dL_dZ_next, W_next, A_prev, m):
        #Compute dL_dA,
        dL_dA = np.dot(W_next.T, dL_dZ_next)
        #Compute dA_dZ i.e the differential of this layer's activation w.r.t this layer's Z
        dA_dZ = self.activation.diff(self.Z)
        #Compute dZ i.e dL_dZ of this layer.
        self.dZ = np.multiply(dL_dA, dA_dZ)
        #Compute dW and dB, with the same shape as W and B respectively
        #(1/m) is included to prevent a grow in loss wih more examples being used.
        self.dW = (1/m)*np.dot(self.dZ, A_prev.T)
        self.dB = (1/m)*(np.sum(self.dZ, axis=1, keepdims=True))

    def out_grad(self, dL_dZ, A_prev, m):
        self.dZ = dL_dZ
        self.dW = (1./m)*(np.dot(self.dZ, A_prev.T))
        self.dB = (1./m)*(np.sum(self.dZ, axis=1, keepdims=True))

    def step(self, lr):
        #lr stands for learning rate.
        self.W = self.W - lr*self.dW
        self.B = self.B - lr*self.dB


###The Genetic Algorithm
#For this project, the gen_length of the first organisms (being hyperparameters of the NN) is 4.
# individual[0] = number of epochs (n_epochs)
# individual[1] = learn rate (lr)
# individual[2] = learn rate decay (lr_decay)
# individual[3] = batch size (batch_size)

#If the initial test is succesful, the following parameters might be added, giving gen_length = 12:
# individual[4] = loss function (lossfn)
# individual[5] = activation function (activation)
# individual[6] = number of layer (layers)
# individual[7] = amount of begin layer nodes (nodes_b)
# individual[8] = amount of first hidden layer nodes (nodes_h1)
# individual[9] = amount of second hidden layer nodes (nodes_h2)
# individual[10] = amount of third hidden layer nodes (nodes_h3)
# individual[11] = amount of final layer nodes (nodes_f)

def F(genome): #The fitness function inputs a certain amount of hyperparameters into the NN_Basic, and outputs its accuracy
    n_epochs_i = int(math.ceil(genome[0])) #Needs to be a whole number number that is not 0 --> integer & ceiling
    if genome[1] >= 1: #Ensuring the learning rate is a value between 0 and 1
        lr_i = genome[1]/100
    elif genome[1] <= 0:
        lr_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        lr_i = genome[1]

    if genome[2] >= 1: #Ensuring the learning rate decay is a value between 0 and 1
        lr_decay_i = genome[2]/100
    elif genome[2] <= 0:
        lr_decay_i = 0.000001 #to avoid /0 error, or get negative logs.
    else:
        lr_decay_i = genome[2]

    if genome[3] <= 0:
        batch_size_i = 1
    else:
        batch_size_i = int(math.ceil(genome[3])) #Needs to be a whole number --> integer & ceil (so it doesn't become 0)

    t_0 = time.perf_counter()
    model = NN_Basic(X.shape[0], Y.shape[0])
    fitness_1 = 1 - acc_level(model, X, Y, X_test, Y_test, model_accuracy, n_epochs_i, batch_size_i, lr_i, lr_decay_i)
    t_f = time.perf_counter()
    Delta_t = (t_f - t_0) * 0.01 #Penalty to equalise acc_level and time's influence
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

def mutate(C, mut_range = 0.5, mut_chance = 0.2): #Arbitrary values
    #mut_range is the range of the amplitude of the mutation is going from -mut_range to mut_range, to ensure both larger & smaller, positive & negatives values for the new mutation.
    #mut_chance is the chance of the mutation actually occuring.
    for i in range(np.size(C)):
        if np.random.uniform(0,1) <= mut_chance: #For a 5% chance, mutate (perform addition) C_i with either 0.95 or 1.05 (slightly increasing or decreasing the value of an allele)
            mut_range = np.random.choice([1-mut_range, 1+mut_range])
            C[i] = C[i] * mut_range
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

def run(N = 10, gen_length = 4, num_generations = 30, Fitness_function = F):
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

#Log 18-03:
#When testing for the fitness of an individual, it is important that the genomes can be translated back into hyperparameters and a NN can be trained.
#To do this, n_epochs ([0]), batch_size ([3]) and all of the additional hyperparameters ([4-11]) have to be translated into integers again.
#Moreover, [1] and [2] need to be a value between 0 and 1, whilst we want to limit other paramaters aswell (e.g. [4] can only be 0 and 1, [6] can only be 2-5, etc)
#--> How do we tell the GA that alleles can only be within a certain range? Afterwards, we can just revert the genes that need it back to integers which can in turn also represent certain functions.
#The difficult part here is to specify for each gene what its range can be.
#Insight: For now, if that value is bigger or smaller than the maximum or minimum, take the maximum or minimum.

#Log 19-03:
#Everything seems to be in place now, including the 'timed' aspect of the fitness (although the penalty needs to be played around with).
#The next step is to add a dataset from the last assignment to see if this NN GA is somewhat efficient, regardless of computation time.
#A.K.A. Debugging time!

best_fitness, best_genome, fitness_his = run()
print(fitness_his)
