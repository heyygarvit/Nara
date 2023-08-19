import sys
sys.path.append('src')
from core.GPU import cp
from optimizers.Adam import Adam

# Define a simple one-layer neural network
class SimpleNN:
    def __init__(self, icput_dim, output_dim):
        self.weights = self.initialize_weights(icput_dim, output_dim)
    
    def initialize_weights(self, icput_dim, output_dim):
        class Weights:
            pass
        weights = Weights()
        weights.data = cp.random.randn(icput_dim, output_dim) * 0.01
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
icput_dim = 2
output_dim = 1
learning_rate = 0.01
epochs = 1000

# Initialize the neural network and the optimizer
model = SimpleNN(icput_dim, output_dim)
optimizer = Adam([model.weights], learning_rate=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(x_train)
    
    # Compute loss and gradient
    loss = cp.mean((y_pred - y_train) ** 2)
    grad_output = 2 * (y_pred - y_train) / y_train.shape[0]
    
    # Backward pass
    model.backward(x_train, grad_output)
    
    # Update weights using Adam optimizer
    optimizer.step()
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Check the updated weights
print("Updated weights:")
print(model.weights.data)

# Check the log data from the optimizer
# print("Optimizer log data:")
# for log in optimizer.log_data:
#     print(log)
