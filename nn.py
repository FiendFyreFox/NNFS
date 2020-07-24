import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
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

# === MODEL ===

class Model:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs + 1):
            
            output = self.forward(X, training=True)

            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            
            if not epoch % print_every:
                print('epoch:', epoch, 'acc:', f'{accuracy:.3f}', 'loss:', f'{loss:.3f}', '(data_loss:', f'{data_loss:.3f}', 'reg_loss:', f'{regularization_loss:.3f})', 'lr:', self.optimizer.current_learning_rate)

        if validation_data is not None:

            X_val, y_val = validation_data

            output = self.forward(X_val, training=False)

            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


    def finalize(self):

        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):

            # If it's the first one, the previously called object will be the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1: # all hidden layers except for last
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else: # last layer before loss function
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output # now the last object from the list

    def backward(self, output, y):

        # First call backward on the loss
        # this provides the initial dvalues
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dvalues as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dvalues)


# === ACTIVATION FUNCTIONS ===

class Activation_Linear:
    
    def forward(self, inputs, training):
        self.input = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Activation_ReLU:
    
    def forward(self, inputs, training):
        
        self.inputs = inputs # remember input values
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable, 
        # let's make a copy of the values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative 
        self.dvalues[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Activation_Softmax:
    #Forward pass
    def forward(self, inputs, training):

        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Sigmoid:

    def forward(self, inputs, training):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dvalues = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# === LAYERS ===

class Layer_Input:

    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.inputs = n_inputs

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):

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

    def forward(self, values, training):
        self.input = values

        if not training:
            self.output = values.copy()
            return

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

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):

        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers:
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

class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        data_loss = 2 * np.mean((y_true - y_pred)**2, axis=-1) # << TODO change this to whatever the correct formula is

        return data_loss

    def backward(self, dvalues, y_true):
        self.dvalues = -(y_true - dvalues)

# === ACCURACY === 

class Accuracy:

    def calculate(self, predictions, y):

        comparisions = self.compare(predictions, y)

        accuracy = np.mean(comparisions)

        return accuracy

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None
    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 500

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):

    def init(self, y):
        pass

    def compare(self, predictions, y):
        return predictions == y

# === Miscellaneous ===

X, y = spiral_data(1000, 3)
X_test, y_test = spiral_data(1000, 3)
#y = y.reshape(-1, 1)
#y_test = y_test.reshape(-1, 1)

model = Model()

model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss and optimizer objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=1e-8),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=800, print_every=100)
'''
def plot_data(myX, myy):
    plt.scatter(myX[:, 0], myX[:, 1], c=myy, s=40, cmap='brg')
    plt.show()

#plot_data(X, y)
dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()

loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(decay=1e-8) # << TODO find out why my results are always different

# Calculate approximate accuracy metric
accuracy_precision = np.std(y) / 500

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    #print(activation2.output[:5])

    sample_losses = loss_function.forward(activation3.output, y)
    data_loss = np.mean(sample_losses)

    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)

    # Overall loss:
    loss = data_loss + regularization_loss


    #calculate accuracy:
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    #print('acc:', accuracy)

    if not epoch % 100:
        print('epoch:', epoch, 'acc:', f'{accuracy:.3f}', 'loss:', f'{loss:.3f}', '(data_loss:', f'{data_loss:.3f}', 'reg_loss:', f'{regularization_loss:.3f})', 'lr:', optimizer.current_learning_rate)

    # backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dvalues)
    dense3.backward(activation3.dvalues)
    activation2.backward(dense3.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
'''
'''
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
'''