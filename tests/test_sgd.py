import sys
sys.path.append('src')
from optimizers.SGD import SGD
from core.GPU import cp

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
x_train = cp.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y_train = cp.array([[2], [3], [4], [5], [6], [7], [8], [9]])

# Hyperparameters
icput_dim = 2
output_dim = 1
learning_rate = 0.001
epochs = 1000
batch_size = 2  # Mini-batch size

# Initialize the neural network and the optimizer
model = SimpleNN(icput_dim, output_dim)
optimizer = SGD([model.weights], learning_rate=learning_rate)

# Training loop
for epoch in range(epochs):
    # Shuffle the dataset for stochasticity
    indices = cp.arange(x_train.shape[0])
    cp.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    for i in range(0, len(x_train), batch_size):
        # Mini-batch data
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        y_pred = model.forward(x_batch)
        
        # Compute loss and gradient
        loss = cp.mean((y_pred - y_batch) ** 2)
        grad_output = 2 * (y_pred - y_batch) / y_batch.shape[0]
        
        # Backward pass
        model.backward(x_batch, grad_output)
        
        # Update weights using SGD optimizer
        optimizer.step()
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Check the updated weights
print("Updated weights:")
print(model.weights.data)



