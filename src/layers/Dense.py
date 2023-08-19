import numpy as np
import sys
sys.path.append('src')
from core.GPU import  USE_GPU, check_memory, get_memory_info, set_device, cp
from core.Tensor import Tensor


class Layer:
    def __init__(self, layers=[]):
        self.layers = layers
        self.parameters = []


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)
    

    def get_parameters(self):
        for layer in self.layers:
            self.parameters.extend(layer.get_parameters())
        return self.parameters

    
class Dense(Layer):
    def __init__(self, input_units, output_units, activation_func=None, init_method="xavier", l1_reg=0.0, l2_reg=0.0):
        super().__init__()

         # Weight Initialization
        if init_method == "xavier":
            self.weights = Tensor(cp.random.randn(input_units, output_units) * cp.sqrt(2. / (input_units + output_units)), requires_grad=True)
        elif init_method == "he":
            self.weights = Tensor(cp.random.randn(input_units, output_units) * cp.sqrt(2. / input_units), requires_grad=True)
        else:
            self.weights = Tensor(cp.random.randn(input_units, output_units) * 0.01, requires_grad=True)

        self.bias = Tensor(cp.zeros((1, output_units)), requires_grad=True)
        self.activation_func = activation_func
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.parameters.extend([self.weights, self.bias])
        # self.get_parameters.append(self.weights)
        # self.get_parameters.append(self.bias)
  
    # Override the get_parameters method for the Dense layer to directly return its parameters
    def get_parameters(self):
        return self.parameters
    
    def forward(self, x):
        # LINEAR TRANSFORMATION
        # z = x.matmul(self.weights) + self.bias
        z = x @ self.weights.data + self.bias.data

            
        # APPLY ACTIVATION FUNCTION(IF ANY)
        if self.activation_func:
            return self.activation_func(z)
        return z


# TRAINING LOOP

#     loss = criterion(predictions, targets)
# if layer.l1_reg:
    # loss += layer.l1_reg * np.sum(np.abs(layer.weights.data))
# if layer.l2_reg:
#     loss += layer.l2_reg * np.sum(layer.weights.data ** 2)

class Sigmoid:
    def __call__(self, x):
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.sigmoid

    def gradient(self):
        return self.sigmoid * (1 - self.sigmoid)

class Tanh:
    def __call__(self, x):
        self.tanh = np.tanh(x)
        return self.tanh

    def gradient(self):
        return 1 - self.tanh**2

class ReLU:
    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def gradient(self):
        return np.where(self.x > 0, 1, 0)
    
class Softmax:
    def __call__(self, x):
        self.x = x
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.softmax

    def cross_entropy(self, y_true):
        # Compute the cross-entropy loss
        n_samples = y_true.shape[0]
        log_likelihood = -np.log(self.softmax[range(n_samples), y_true])
        loss = np.sum(log_likelihood) / n_samples
        return loss

    def delta_cross_entropy(self, y_true):
        # Compute the gradient of the cross-entropy loss with respect to the input of softmax
        n_samples = y_true.shape[0]
        y_pred = self.softmax.copy()
        y_pred[range(n_samples), y_true] -= 1
        y_pred /= n_samples
        return y_pred

class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        self.train_mode = True
        self.parameters.append(self.gamma)
        self.parameters.append(self.beta)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def forward(self, x):
        if self.train_mode:
            # Compute batch statistics
            mean = np.mean(x.data, axis=0)
            var = np.var(x.data, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            # Use running statistics for inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out = self.gamma.data * x_norm + self.beta.data
        return Tensor(out)


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.train_mode = True

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def forward(self, x):
        if self.train_mode:
            self.mask = np.random.binomial(1, 1-self.p, size=x.data.shape) / (1-self.p)
            return x * self.mask
        return x

class LayerNorm(Layer):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.eps = eps
        self.parameters.append(self.gamma)
        self.parameters.append(self.beta)

    def forward(self, x):
        mean = np.mean(x.data, axis=1, keepdims=True)
        var = np.var(x.data, axis=1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out = self.gamma.data * x_norm + self.beta.data
        return Tensor(out)
    

# Parameter Constraints
def constrain_weights(weights, min_val=-1, max_val=1):
    # Constrain weights to be within a specified range
    return np.clip(weights, min_val, max_val)

def optimized_matmul(A, B):
    # Placeholder for an optimized matrix multiplication function
    # This can be replaced with a GPU-accelerated version or any other optimized version
    return np.matmul(A, B)