import os
# import numpy as cp
import logging
import json
import sys
sys.path.append('src')
from core.GPU import cp


logging.basicConfig(level=logging.INFO)

class ModelSaver:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def save(self, path, epoch=None, prefix=""):
        if not os.path.exists(path):
            os.makedirs(path)

        # Convert model weights to NumPy arrays and save
        weights = {}
        if hasattr(self.model, 'named_parameters'):
            for name, param in self.model.named_parameters():
                weights[name] = cp.asarray(param).tolist()  # Convert to list
        else:
            raise AttributeError("Model does not have named_parameters method.")

        # save_file = os.path.join(path, f'{prefix}weights.cpy')
        save_file = os.path.join(path, f'{prefix}weights.npy')

        cp.save(save_file, weights)
        logging.info(f"Model weights saved to {save_file}")

        # Save optimizer state if provided
        if self.optimizer:
            optimizer_state = {}
            if hasattr(self.optimizer, 'state_dict'):
                for key, value in self.optimizer.state_dict().items():
                    optimizer_state[key] = cp.asarray(value).tolist()  # Convert to list
            else:
                raise AttributeError("Optimizer does not have state_dict method.")
            
            # optimizer_file = os.path.join(path, f'{prefix}optimizer.cpy')
            optimizer_file = os.path.join(path, f'{prefix}optimizer.npy')

            cp.save(optimizer_file, optimizer_state)
            logging.info(f"Optimizer state saved to {optimizer_file}")

        # Save epoch number if provided
        metadata = {}
        if epoch is not None:
            metadata['epoch'] = epoch
        metadata_file = os.path.join(path, f'{prefix}metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        logging.info(f"Metadata saved to {metadata_file}")

        return "Model saved successfully."

    def load(self, path, prefix=""):
        metadata = {}
        # Load model weights
        # weights_file = os.path.join(path, f'{prefix}weights.cpy')
        weights_file = os.path.join(path, f'{prefix}weights.npy')

        if os.path.exists(weights_file):
            weights = cp.load(weights_file, allow_pickle=True).item()
            if hasattr(self.model, 'named_parameters'):
                for name, param in self.model.named_parameters():
                    param[:] = weights[name]
            else:
                raise AttributeError("Model does not have named_parameters method.")
        else:
            raise FileNotFoundError(f"{weights_file} not found.")

        # Load optimizer state if provided
        if self.optimizer:
            optimizer_file = os.path.join(path, f'{prefix}optimizer.npy')
            if os.path.exists(optimizer_file):
                optimizer_state = cp.load(optimizer_file, allow_pickle=True).item()
                if hasattr(self.optimizer, 'load_state_dict'):
                    self.optimizer.load_state_dict(optimizer_state)
                else:
                    raise AttributeError("Optimizer does not have load_state_dict method.")
            else:
                raise FileNotFoundError(f"{optimizer_file} not found.")

        # Load metadata
        metadata_file = os.path.join(path, f'{prefix}metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        logging.info(f"Model loaded from {path}")
        return metadata
