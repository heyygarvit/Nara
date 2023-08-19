# import numpy as cp
import sys
sys.path.append('src')
from core.GPU import cp
from optimizers.GradientDescent import GradientDescent

# Define a simple one-layer neural network
class SimpleNN:
    def __init__(self, input_dim, output_dim):
        self.weights = self.initialize_weights(input_dim, output_dim)
    
    def initialize_weights(self, input_dim, output_dim):
        class Weights:
            pass
        weights = Weights()
        weights.data = cp.random.randn(input_dim, output_dim) * 0.01
        weights.grad = cp.zeros_like(weights.data)
        return weights
    
    def forward(self, x):
        return cp.dot(x, self.weights.data)
    
    def backward(self, x, grad_output):
        self.weights.grad = cp.dot(x.T, grad_output)
        return self.weights.grad

# Create a toy dataset
x_train = cp.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = cp.array([[2], [3], [4], [5]])

# Hyperparameters
input_dim = 2
output_dim = 1
# learning_rate = 0.01
epochs = 1000

# Initialize the neural network and the optimizer
model = SimpleNN(input_dim, output_dim)


optimizer = GradientDescent(learning_rate=0.01)


# Training loop
# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(x_train)
    
    # Compute loss and gradient
    loss = cp.mean((y_pred - y_train) ** 2)
    grad_output = 2 * (y_pred - y_train) / y_train.shape[0]
    
    # Backward pass
    model.backward(x_train, grad_output)
    
    # Update weights using GradientDescent optimizer
    updated_weights = optimizer.step({'data': model.weights.grad}, {'data': model.weights.data})
    model.weights.data = updated_weights['data']
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Check the updated weights
print("Updated weights:")
print(model.weights.data)

