import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import random

nnfs.init()

np.random.seed(0)
random.seed(0)

def create_data(n, k):
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

class Activation_Sigmoid:

    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dvalues = dvalues * (1 - self.output) * self.output

# === LAYERS ===

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.inputs = n_inputs

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):

        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 weights
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 biases
        if self.bias_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)

class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, values):
        self.input = values

        self.binary_mask = np.random.binomial(1, self.rate, size=values.shape) / self.rate
        # Apply mask to output values
        self.output = values * self.binary_mask

    def backward(self, dvalues):
        self.dvalues = dvalues * self.binary_mask

# === OPTIMIZERS ===

class Optimizer_SGD:

    # Initialize optimizer - set settings, 
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.): 
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations)) # <<TODO change this to self.learning rate once the book is fixed

    # Update parameters
    def update_params(self, layer):

        # If a layer does not already store past momentum, create arrays for that purpose
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # If we use momentum
        if self.momentum:

            weight_updates = ((self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights))
            layer.weight_momentums = weight_updates

            bias_updates = ((self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases))
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:

    # Initialize optimizer - set settings, 
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7): 
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations)) # <<TODO change this to self.learning rate once the book is fixed

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        weight_updates = -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        bias_updates = -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:

    # Initialize optimizer - set settings, 
    # learning rate of 0.001 is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9): 
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations)) # <<TODO change this to self.learning rate once the book is fixed

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        weight_updates = -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        bias_updates = -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:

    # Initialize optimizer - set settings, 
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999): 
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations)) # <<TODO change this to self.learning rate once the book is fixed

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache/momentum arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update momentum
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        weight_updates = -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        bias_updates = -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

# === LOSS FUNCTIONS ===

# Common loss class
class Loss:

    def regularization_loss(self, layer):

        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        #L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

         # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        #L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

        
class Loss_CategoricalCrossentropy(Loss):

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

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return sample_losses

    def backward(self, dvalues, y_true):

        # clip both sides
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dvalues = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))

# === Miscellaneous ===

X, y = spiral_data(100, 2)
y = y.reshape(-1, 1)

def plot_data(myX, myy):
    plt.scatter(myX[:, 0], myX[:, 1], c=myy, s=40, cmap='brg')
    plt.show()

plot_data(X, y)
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
#dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(64, 1)
activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()
optimizer = Optimizer_Adam(decay=1e-8) # << TODO find out why my results are always different

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    #print(activation2.output[:5])

    sample_losses = loss_function.forward(activation2.output, y)
    data_loss = np.mean(sample_losses)

    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    # Overall loss:
    loss = data_loss + regularization_loss


    #calculate accuracy:
    predictions = (activation2.output > 0.5) * 1  # calculate values along first axis
    accuracy = np.mean(predictions==y)

    #print('acc:', accuracy)

    if not epoch % 1000:
        print('epoch:', epoch, 'acc:', f'{accuracy:.3f}', 'loss:', f'{loss:.3f}', '(data_loss:', f'{data_loss:.3f}', 'reg_loss:', f'{regularization_loss:.3f})', 'lr:', optimizer.current_learning_rate)

    # backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(100, 2)
y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)


sample_losses = loss_function.forward(activation2.output, y_test)
loss = np.mean(sample_losses)


#calculate accuracy:
predictions = (activation2.output > 0.5) * 1  # calculate values along first axis
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')