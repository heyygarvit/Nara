
import logging
import sys
sys.path.append('src')
from core.GPU import cp, USE_GPU, check_memory, set_device, convert_dtype, fallback_mechanism

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, use_gpu=USE_GPU, device_id=0, augment_data=False, dynamic_batch_resizing=False, sequence_data=False, collate_fn=None, MAX_BATCH_SIZE = 512 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_gpu = use_gpu
        self.augment_data = augment_data
        self.dynamic_batch_resizing = dynamic_batch_resizing
        self.sequence_data = sequence_data
        self.collate_fn = collate_fn
        self.maxbatchsize = MAX_BATCH_SIZE
        self.index = 0
        if self.use_gpu:
            set_device(device_id)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset[0]):
            self.index = 0
            if self.shuffle:
                self._shuffle_dataset()
            raise StopIteration

        try:
            batch_X = self.dataset[0][self.index:self.index+self.batch_size]
            batch_y = self.dataset[1][self.index:self.index+self.batch_size]
            
            # Data Augmentation
            if self.augment_data:
                batch_X = self._augment_batch(batch_X)
            
            # Sequence Data Padding
            if self.sequence_data:
                batch_X = self._pad_sequences(batch_X)
            
            # Custom Collate Function
            if self.collate_fn:
                batch_X, batch_y = self.collate_fn(batch_X, batch_y)
            
        except Exception as e:
            logging.error(f"Error fetching batch data: {e}")
            raise

        self.index += self.batch_size

        # Dynamic Batch Resizing with a maximum limit
         # You can adjust this value based on your needs
        if self.dynamic_batch_resizing:
            self.batch_size = min(self.batch_size + 1, self.maxbatchsize , len(self.dataset[0]))

        # Transfer to GPU if required
        if self.use_gpu:
            try:
                check_memory(batch_X.nbytes + batch_y.nbytes)
                batch_X = convert_dtype(batch_X, dtype=cp.float32)
                batch_y = convert_dtype(batch_y, dtype=cp.float32)
                batch_X = cp.array(batch_X)
                batch_y = cp.array(batch_y)
            except Exception as e:
                logging.warning(f"GPU operation failed: {e}. Using CPU fallback.")
                batch_X, batch_y = fallback_mechanism((batch_X, batch_y))

        return batch_X, batch_y

    def _shuffle_dataset(self):
        """Shuffle the dataset."""
        permutation = cp.random.permutation(len(self.dataset[0]))
        shuffled_X = [self.dataset[0][i] for i in permutation]
        shuffled_y = [self.dataset[1][i] for i in permutation]
        self.dataset = (shuffled_X, shuffled_y)

    def _shuffle_dataset(self):
        """Shuffle the dataset."""
        permutation = cp.random.permutation(len(self.dataset[0]))
        shuffled_X = cp.array([self.dataset[0][i] for i in permutation])
        shuffled_y = cp.array([self.dataset[1][i] for i in permutation])
        self.dataset = (shuffled_X, shuffled_y)


    
    def _augment_batch(self, batch_X):
    # Simple random horizontal flip for demonstration
        flip_indices = cp.random.choice([True, False], size=len(batch_X))
        flip_int_indices = cp.where(flip_indices)[0]  # Convert boolean indices to integer indices
        
        for idx in flip_int_indices:
            batch_X[idx] = cp.flip(batch_X[idx], axis=1)  # Assuming batch_X is of shape (batch_size, height, width). If not, adjust the axis accordingly.
        
        return batch_X


    def _pad_sequences(self, batch_X):
        # Generalized padding for sequences of varying dimensions
        max_len = max([len(seq) for seq in batch_X])
        padded_shape = (len(batch_X), max_len) + batch_X[0].shape[1:]
        padded_batch = cp.zeros(padded_shape)
        for i, seq in enumerate(batch_X):
            padded_batch[i, :len(seq)] = seq
        return padded_batch

