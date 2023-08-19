import time
import numpy as np
# from tqdm import tqdm  # For progress bars
import sys
sys.path.append("src")
from core.GPU import cp
from core.Tensor import Tensor


class TrainingError(Exception):
    pass
    

def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=1, patience=5, scheduler=None, callbacks=None):
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if not train_loader or not val_loader:
        raise TrainingError("Data loaders cannot be empty")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0 
        batch_count = 0

        #TRAINING WITH A PROGRESS BAR
        for batch_data, batch_labels in train_loader:
            batch_count +=1
            batch_data = cp.array(batch_data)
            batch_labels = cp.array(batch_labels)
            if not isinstance(batch_data, cp.ndarray) or not isinstance(batch_labels, cp.ndarray):
                raise TrainingError("Batch data or labels are not numpy arrays")
            if batch_data.size == 0 or batch_labels.size == 0:
                raise TrainingError("Batch data or labels are missing.")
            predictions = model(batch_data)
            batch_labels = Tensor(batch_labels, requires_grad=True)  # Ensure batch_labels is a Tensor
            loss = loss_fn(predictions, batch_labels)
            # print("Loss value:", loss.data)
            # print("Gradient of predictions before backward:", predictions.grad)
            # print("Gradient of batch_labels before backward:", batch_labels.grad)

            zero_gradients(model)
            # Tensor.backward(loss)
            loss.backward(1.0)
            # print("LOSS",loss.backward(1.0))
            # print("Gradient of predictions after backward:", predictions.grad)
            # print("Gradient of batch_labels after backward:", batch_labels.grad)
            # print("Gradient for weights:", model.weights.grad)
            # print("Gradient for bias:", model.bias.grad)
            optimizer.step()
            epoch_loss += loss.data
        avg_train_loss = epoch_loss / batch_count
        train_history.append(avg_train_loss)

        #VALIDATION WITH A PROGRESS BAR
        val_batch_count = 0
        val_loss = 0.0
        for batch_data, batch_labels in val_loader:
            val_batch_count += 1
            predictions = model(batch_data)
            loss = loss_fn(predictions, batch_labels)
            val_loss += loss.data
        avg_val_loss = val_loss / val_batch_count
        val_history.append(avg_val_loss)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {elapsed_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # model.save_weights('best_model_weights.dat')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler:
            scheduler.step(avg_val_loss)

        if epochs_without_improvement >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

        if callbacks:
            for callback in callbacks:
                callback(epoch, model, avg_train_loss, avg_val_loss)

    return train_history, val_history


# ADDITIONAL UTILITY FUNCTIONS
def load_model_weights(model, path):
    try:
        # ASSUMING MODAL HAS A METHOD TO LOAD WEIGHTS
        model.load_weights(path)
    except Exception as e:
        raise TrainingError(f"Error loading model weights: {e}")
    
def save_model_weights(model, path):
    try:
        # ASSUMING MODAL HAS A METHOD TO SAVE WEIGHTS
        model.save_weights(path)
    except Exception as e:
        raise TrainingError(f"Error saving model weights: {e}")
    
def zero_gradients(model):
    """Zero out the gradients of all parameters in the model."""
    for param in model.get_parameters():
        Tensor.zero_grad(param)
        
     