# import numpy as np
# import sys
# sys.path.append('src')
# from core.Training import load_model_weights, save_model_weights, train



# # Define a simple linear regression model
# class LinearRegression:
#     def __init__(self, input_dim):
#         self.weights = Tensor(np.random.randn(input_dim, 1))
#         self.bias = Tensor(np.random.randn(1))

#     def __call__(self, x):
#         return x @ self.weights + self.bias

#     def parameters(self):
#         return [self.weights, self.bias]

#     def save_weights(self, path):
#         np.savez(path, weights=self.weights.data, bias=self.bias.data)

#     def load_weights(self, path):
#         data = np.load(path)
#         self.weights.data = data['weights']
#         self.bias.data = data['bias']

# # Define a mean squared error loss function
# # def mean_squared_error(predictions, targets):
# #     return ((predictions - targets) ** 2).mean()

# # Generate synthetic data
# np.random.seed(42)
# X_train = np.random.randn(100, 1)
# y_train = 3 * X_train + 2 + 0.1 * np.random.randn(100, 1)

# X_val = np.random.randn(50, 1)
# y_val = 3 * X_val + 2 + 0.1 * np.random.randn(50, 1)

# train_loader = [(X_train[i:i+10], y_train[i:i+10]) for i in range(0, 100, 10)]
# val_loader = [(X_val[i:i+10], y_val[i:i+10]) for i in range(0, 50, 10)]

# # Initialize model, loss function, and optimizer
# model = LinearRegression(input_dim=1)
# optimizer = GradientDescent(learning_rate=0.01)
# mean_squared_error = Metrics.mean_squared_error
# # optimizer = lambda: None  # Dummy optimizer for demonstration
# # optimizer.step = lambda: None  # Dummy step function

# # Train the model
# train_history, val_history = train(model, train_loader, val_loader, mean_squared_error, optimizer, num_epochs=5)

# # Save and load model weights
# save_model_weights(model, 'model_weights.npz')
# load_model_weights(model, 'model_weights.npz')

# print("Training complete!")



# import numpy as np
# import sys
# sys.path.append("src")
# from core.Tensor import Tensor
# from core.Training import train
# from core.Tensor import Tensor
# from optimizers.SGD import SGD
# from optimizers.GradientDescent import GradientDescent
# from utils.EvalutionMetrics import Metrics

# # Define a simple linear regression model
# class LinearRegression:
#     def __init__(self, input_dim):
#         self.weights = Tensor(np.random.randn(input_dim, 1), requires_grad=True)
#         self.bias = Tensor(np.zeros((1,)), requires_grad=True)
        
    
#     # def __call__(self, x):
#     #     x = np.reshape(x, (-1, self.weights.data.shape[0]))  # Ensure x is a 2D array
#     #     return Tensor(x @ self.weights.data + self.bias.data, requires_grad=True)  # Ensure the output is a Tensor

#     # def __call__(self, x):
#     #     x = np.reshape(x, (-1, self.weights.data.shape[0]))  # Ensure x is a 2D array
#     #     return x @ self.weights.data + self.bias.data
    
#     # def __call__(self, x):
#     #     x = Tensor._ensure_tensor(x)  # Ensure x is a Tensor
#     #     x = x.data.reshape((-1, self.weights.data.shape[0]))  # Ensure x is a 2D Tensor
#     #     return x @ self.weights + self.bias
    
#     # def __call__(self, x):
#     #     x = Tensor._ensure_tensor(x)  # Ensure x is a Tensor
#     #     x.data = np.array(x.data)
#     #     x = x.data.reshape((-1, self.weights.data.shape[0]))  # Ensure x is a 2D Tensor
#     #     print("Shape of x:", x.data.shape)
#     #     print("Type of x data:", type(x.data))
#     #     print("Values of x:", x.data)
        
#     #     print("Shape of weights:", self.weights.data.shape)
#     #     print("Type of weights data:", type(self.weights.data))
#     #     print("Values of weights:", self.weights.data)
        
#     #     # Try matrix multiplication using numpy
#     #     try:
#     #         multiplied = np.dot(x.data, self.weights.data)
#     #     except Exception as e:
#     #         print("Error during multiplication:", e)
        
#     #     return multiplied + self.bias.data

#     def __call__(self, x):
#         x = Tensor._ensure_tensor(x)  # Ensure x is a Tensor
#         x.data = np.array(x.data)  # Convert memoryview to numpy array if needed
#         x = x.data.reshape((-1, self.weights.data.shape[0]))  # Ensure x is a 2D Tensor
        
#         # multiplied = x @ self.weights
#         multiplied = Tensor(np.dot(x.data, self.weights.data))

#         result = multiplied + self.bias
        
#         # Ensure the result is a Tensor object
#         if not isinstance(result, Tensor):
#             result = Tensor(result)
        
#         return result

    
#     def parameters(self):
#         return [self.weights, self.bias]


# # Define a simple mean squared error loss function
# # def mean_squared_error(predictions, targets):
# #     diff = predictions - targets
# #     return (diff * diff).mean()

# # Generate some dummy data for testing
# np.random.seed(42)
# x_train = np.random.randn(100, 1)
# y_train = 2 * x_train + 3 + 0.1 * np.random.randn(100, 1)

# x_val = np.random.randn(20, 1)
# y_val = 2 * x_val + 3 + 0.1 * np.random.randn(20, 1)

# train_loader = list(zip(x_train, y_train))
# val_loader = list(zip(x_val, y_val))

# # Define a simple optimizer (e.g., SGD)
# # class SGD:
# #     def __init__(self, parameters, lr=0.01):
# #         self.parameters = parameters
# #         self.lr = lr

# #     def step(self):
# #         for param in self.parameters:
# #             param.data -= self.lr * param.grad.data

# # Instantiate the model, optimizer and train
# model = LinearRegression(input_dim=1)
# optimizer = SGD(model.parameters(), learning_rate=0.001)
# # optimizer = GradientDescent(learning_rate=0.01)
# # optimizer = GradientDescent(learning_rate=0.01)
# # optimizer.step({'data': model.weights.grad}, {'data': model.weights.data})

# Metrics.mean_squared_error

# train_history, val_history = train(model, train_loader, val_loader, Metrics.mean_squared_error, optimizer, num_epochs=5)

# print("Training Losses:", train_history)
# print("Validation Losses:", val_history)





import numpy as np
import sys
sys.path.append("src")
from core.Tensor import Tensor
from core.Training import train
from core.Tensor import Tensor
from optimizers.SGD import SGD
from optimizers.GradientDescent import GradientDescent
from utils.EvalutionMetrics import Metrics

# Define a simple linear regression model
class LinearRegression:
    def __init__(self, input_dim):
        self.weights = Tensor(np.random.randn(input_dim, 1), requires_grad=True)
        self.bias = Tensor(np.zeros((1,)), requires_grad=True)
        
   
    def __call__(self, x):
        x = Tensor._ensure_tensor(x)  # Ensure x is a Tensor
        x.data = np.array(x.data)  # Convert memoryview to numpy array if needed
        x = x.data.reshape((-1, self.weights.data.shape[0]))  # Ensure x is a 2D Tensor
        
        # multiplied = x @ self.weights
        multiplied = Tensor(np.dot(x.data, self.weights.data))

        result = multiplied + self.bias
        
        # Ensure the result is a Tensor object
        if not isinstance(result, Tensor):
            result = Tensor(result)
        
        return result

    
    def parameters(self):
        return [self.weights, self.bias]



np.random.seed(42)
x_train = np.random.randn(100, 1)
y_train = 2 * x_train + 3 + 0.1 * np.random.randn(100, 1)

x_val = np.random.randn(20, 1)
y_val = 2 * x_val + 3 + 0.1 * np.random.randn(20, 1)

train_loader = list(zip(x_train, y_train))
val_loader = list(zip(x_val, y_val))



# Instantiate the model, optimizer and train
model = LinearRegression(input_dim=1)
optimizer = SGD(model.parameters(), learning_rate=0.001)


Metrics.mean_squared_error

train_history, val_history = train(model, train_loader, val_loader, Metrics.mean_squared_error, optimizer, num_epochs=100)

print("Training Losses:", train_history)
print("Validation Losses:", val_history)
