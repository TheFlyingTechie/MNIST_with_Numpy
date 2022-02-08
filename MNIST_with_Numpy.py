from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import time

#Neural Network class that is going to be trained
class NeuralNetwork():
    def __init__(self, sizes, epochs=10, lr=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        #Parameters are the weights, saved in this dictionary
        self.params = self.initialization()
        
    def relu(self, x, derivative=False):
        #relu activation function
        if derivative:
            X = x[:]
            X[X <= 0] = 0
            X[X > 0] = 1
            return X
        return np.maximum(0, x)
    
    def softmax(self, x):
        #Softmax activation function, softmax is numerically stable with large exponentials
        #This is the only implemented function for the output layer. Don't use it for anything else
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    
    def tanh(self, x, derivative=False):
        #Tanh activation function.
        tanh = (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
        if derivative:
            return 1 - np.power(tanh, 2)
        return tanh
    
    def sigmoid(self, x, derivative=False):
        #Sigmoid activation function
        if derivative:
            return x * (1.0 - x)
        return 1 / (1+ np.exp(-x))
    
    def initialization(self):
        #number of nodes for each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]
        
        #calculating parameters
        params = {'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
                  'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
                  'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)}
        return params
    
    def forward_pass(self, x_train):
        #forward propagation function
        params = self.params
        
        #input layer activation
        params['A0'] = x_train
        
        #input layer to hidden layer one
        params['Z1'] = np.dot(params['W1'], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])            #This can be changed
        
        #hidden layer one to hidden layer two
        
        params['Z2'] = np.dot(params['W2'], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])            #So can this
        
        #hidden layer two to the output layer
        params['Z3'] = np.dot(params['W3'], params['A2'])
        params['A3'] = self.softmax(params['Z3'])         #Try to keep this the same, but change if you like
        
        return params['A3']
    
    def backward_pass(self, y_train, output):
        #Backpropagation function, which claculates the updates to the neural network
        #There may be errors because of the dot and multipy functions on large arrays
        
        params = self.params
        change_w = {}
        
        #Calculate from back to front
        #W3 update
        error = output - y_train
        change_w['W3'] = np.dot(error, params['A3'])
        
        #W2 update
        error = np.multiply(np.dot(params['W3'].T, error), self.sigmoid(params['Z2'], derivative=True))  #Change here
        change_w['W2'] = np.dot(error, params['A2'])
        
        #W1 update
        error = np.multiply(np.dot(params['W2'].T, error), self.sigmoid(params['Z1'], derivative=True))  #And here
        change_w['W1'] = np.dot(error, params['A1'])
        return change_w
    
    def update_network_params(self, change_w):
        #Updating network parameters
        
        for key, value in change_w.items():
            for w_arr in self.params[key]:
                w_arr -= self.lr * value
                
    def compute_accuracy(self, x_val, y_val):
        #Calculates the accuracy for the network
        predictions = []
        
        for x, y in zip(x_val,y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred==y)
            
        summed = sum(pred for pred in predictions) / 100.0
        return np.average(summed)
    
    def train(self, x_train, y_train, x_val, y_val):
        #function that trains the neural network
        start_time = time.time()
        for iteration in range(self.epochs):
            epoch_start_time = time.time()
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                
                change_w = self.backward_pass(y,output)
                self.update_network_params(change_w)
            accuracy = self.compute_accuracy(x_val, y_val)
            print(f'Epoch: {iteration+1}/{self.epochs}, Time spent for epoch: {time.time()-epoch_start_time}s, Total time: {time.time()-start_time} Accuracy: {accuracy}')

#Driver code
image_size = 28*28
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.array(x_train.reshape(x_train.shape[0], image_size))
x_train = x_train[:(len(x_train)-1)//2]
x_test = np.array(x_test.reshape(x_test.shape[0], image_size))

y_train = np.array(to_categorical(y_train, num_classes))
y_train = y_train[:(len(y_train)-1)//2]
y_test = np.array(to_categorical(y_test, num_classes))

network = NeuralNetwork(sizes=[784, 128, 64, 10])
network.train(x_train, y_train, x_test, y_test)
