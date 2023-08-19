import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys
sys.path.append('src')
from layers.Dense import Dense, ReLU, Softmax, Layer
from core.Tensor import Tensor
from core.Training import train
from optimizers.SGD import SGD
from utils.EvalutionMetrics import Metrics 
# Load the Iris dataset
data = pd.read_csv('assets/dataset/Iris.csv.xls')

# Preprocess the data
X = data.iloc[:, 1:5].values  # Feature columns
y = data.iloc[:, 5].values    # Target column

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
input_units = X_train.shape[1]
hidden_units = 5
output_units = y_train.shape[1]

train_loader = list(zip(X_train , y_train))
val_loader = list(zip(X_val , y_val))

model = Layer([
    Dense(input_units=input_units, output_units=hidden_units, activation_func=ReLU()),
    Dense(input_units=hidden_units, output_units=output_units, activation_func=Softmax())
])
optimizer = SGD(model.get_parameters(), learning_rate=0.001)

# Define the training loop
learning_rate = 0.01
num_epochs = 100

Metrics.mean_squared_error
train_history, val_history = train(model, train_loader, val_loader, Metrics.mean_squared_error, optimizer, num_epochs=100)

print("Training Losses:", train_history)
print("Validation Losses:", val_history)


# for epoch in range(num_epochs):
#     # Forward pass
#     a = Tensor(X_train)
#     for layer in model:
#         a = layer.forward(a)
    

#     # Compute the loss
#     softmax = Softmax()
#     loss = softmax.cross_entropy(np.argmax(y_train, axis=1))

#     # TODO: Implement backward pass and optimization
    
#     # Print the loss
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss}')

# # TODO: Evaluate the model on the validation set
