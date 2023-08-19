import sys
sys.path.append('src')
from utils.ModelSaver import ModelSaver
from core.GPU import cp

# Mock classes for testing
class MockModel:
    def __init__(self):
        self.param1 = cp.array([1, 2, 3])
        self.param2 = cp.array([4, 5, 6])

    def named_parameters(self):
        return [("param1", self.param1), ("param2", self.param2)]

class MockOptimizer:
    def __init__(self):
        self.state = {"lr": 0.001, "momentum": 0.9}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state

# Create mock model and optimizer
model = MockModel()
optimizer = MockOptimizer()

# Initialize ModelSaver
saver = ModelSaver(model, optimizer)


# Save model and optimizer
save_path = "saved_model"
saver.save(save_path, epoch=5)

# Modify model and optimizer to check if loading works
model.param1 = cp.array([0, 0, 0])
model.param2 = cp.array([0, 0, 0])
optimizer.state = {}

# Load model and optimizer
metadata = saver.load(save_path)

# Check if model and optimizer are correctly loaded
assert cp.array_equal(model.param1, [1, 2, 3])
assert cp.array_equal(model.param2, [4, 5, 6])
assert optimizer.state["lr"] == 0.001
assert optimizer.state["momentum"] == 0.9
assert metadata["epoch"] == 5

print("Model and optimizer loaded successfully!")
