import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

np.random.seed(0)

def create_data(n, k): # using this for now to ensure that my results are the same as yours
    X = np.zeros((n*k, 2))  # data matrix (each row = single example)
    y = np.zeros(n*k, dtype='uint8')  # class labels
    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)  # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y

# === ACTIVATION FUNCTIONS ===

class Activation_ReLU:
    
    def forward(self, inputs):
        
        self.inputs = inputs # remember input values
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable, 
        # let's make a copy of the values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative 
        self.dvalues[self.inputs <= 0] = 0


class Activation_Softmax:
    #Forward pass
    def forward(self, inputs):

        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()

# === LAYERS ===

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.inputs = n_inputs

    def forward(self, inputs):

        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)

# === OPTIMIZERS ===

class Optimizer_SGD:

    # Initialize optimizer - set settings, 
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1): # <<< setting this to 1 causes the accuracy to remain exactly 1/3, but a lower value becomes much more accurate.
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# === MISCELLANEOUS ===

class Loss_CategoricalCrossentropy:

    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            y_pred_clipped = y_pred_clipped[range(samples), y_true]

        # Losses
        #print(y_pred[:5])
        negative_log_likelihoods = -np.log(y_pred_clipped)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples

X, y = create_data(100, 3)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
#plt.show()



dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    #print(activation2.output[:5])

    loss = loss_function.forward(activation2.output, y)

    #rint('loss:', loss)

    #calculate accuracy:
    predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
    accuracy = np.mean(predictions==y)

    #print('acc:', accuracy)

    if not epoch % 100:
        print('epoch:', epoch, 'acc:', accuracy, 'loss:', loss)

    # backward pass!
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)











'''
# helper variables
lowest_loss = 99999
best_dense1_weights = dense1.weights
best_dense1_biases = dense1.biases
best_dense2_weights = dense2.weights
best_dense2_biases = dense2.biases



# Forward pass
x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias

# Multiplying inputs by weights
wx0 = x[0] * w[0]
wx1 = x[1] * w[1]
wx2 = x[2] * w[2]

# Adding
s = wx0 + wx1 + wx2 + b

# ReLU
y = max(s, 0)  # we already described that with ReLU activation function description
print(y)

dy = (1 if s > 0 else 0)

dwx0 = 1 * dy 
dwx1 = 1 * dy
dwx2 = 1 * dy
db = 1 * dy

dw0 = (1 if s > 0 else 0) * x[0]
dw1 = (1 if s > 0 else 0) * x[1]
dw2 = (1 if s > 0 else 0) * x[2]
dx0 = (1 if s > 0 else 0) * w[0]
dx1 = (1 if s > 0 else 0) * w[1]
dx2 = (1 if s > 0 else 0) * w[2]

dx = [dx0, dx1, dx2]  # gradients on inputs
dw = [dw0, dw1, dw2]  # gradients on weights
db  # gradient on bias...just 1 bias here.

print(dx, dw)
'''