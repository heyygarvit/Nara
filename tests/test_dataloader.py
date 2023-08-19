import numpy as np
import sys
sys.path.append('src')
from utils.DataLoader import DataLoader

# Mocking the GPU functions for the test
class MockGPU:
    def random(self):
        return np.random

    def zeros(self, shape):
        return np.zeros(shape)

    def array(self, data):
        return np.array(data)

    def flip(self, data, axis):
        return np.flip(data, axis)

    def where(self, condition):
        return np.where(condition)

    def random(self):
        return np.random

cp = MockGPU()
USE_GPU = False

def set_device(device_id):
    pass

def check_memory(size):
    pass

def convert_dtype(data, dtype):
    return data.astype(dtype)

def fallback_Mechanisms(data):
    return data

# Sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y = np.array([[2], [3], [4], [5], [6], [7], [8], [9]])

dataset = (X, y)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, use_gpu=USE_GPU)

# Test DataLoader
for epoch in range(2):  # 2 epochs for demonstration
    print(f"Epoch {epoch + 1}")
    for batch_X, batch_y in dataloader:
        print("Batch X:")
        print(batch_X)
        print("Batch y:")
        print(batch_y)
    print("="*20)
