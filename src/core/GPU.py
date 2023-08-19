import logging
logging.basicConfig(level=logging.WARNING)

try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    import numpy as np
    cp = np #ALIAS CP TO NUMPY
    USE_GPU = False

#DEVICE INFORMATION
def get_device_info():
    if USE_GPU:
        return cp.cuda.runtime.getDeviceProperties(0)
    else:
        return "Running in CPU-only mode."
    
#MEMORY MONITORING
def get_memory_info():
    if USE_GPU:
        memory_info = cp.cuda.runtime.memGetInfo()
        return {"free": memory_info[0], "total": memory_info[1]}
    else:
        return "Memory monitoring is available only in GPU mode."
    

#CONTEXT MANAGEMENT
def set_device(device_id):
    if USE_GPU:
        cp.cuda.Device(device_id).use()


#ADVANCE ERROR HANDLING
def check_memory(size_needed):
    if USE_GPU:
        free_memory = get_memory_info()["free"]
        if size_needed > free_memory:
            raise MemoryError("Not enough GPU memory available.")
        
        
#DATA TYPE HANDLING
def convert_dtype(tensor, dtype):
    return cp.asarray(tensor, dtype=dtype)


#LOGGING AND DEBUGGING
logging.basicConfig(level=logging.INFO)


#VERISON CHECK
def check_version():
    if USE_GPU:
        if cp.__version__ < 'YOUR_MINIMUM_VERSION':
            logging.warning("Your cuPy version is Outdated. Please Update")


#FALLBACK MECHANISM
def fallback_mechanism(data, use_gpu = True):
    try: 
        if use_gpu and USE_GPU:
            check_memory(len(data)*8)  #ASSUMING FLOAT64
            tensor = cp.array(data)
            result = cp.sum(tensor)
            return result
        else:
            raise Exception("Switch to CPU")
    except Exception as e:
        logging.warning(f"GPU operation failed due to: {str(e)}. Switching to CPU.")
        tensor = np.array()
        tensor = np.array(data)
        result = np.sum(tensor)
        return result 
    

#MULTI-GPU SUPPORT
def set_available_gpus(gpu_ids):
    if USE_GPU:
        cp.cuda.runtime.setDeviceMask(sum([1 << i for i in gpu_ids]))


#ASYNCHRONOUS OPERATIONS
def create_stream():
    if USE_GPU:
        return cp.cuda.stream(non_blocking= True)
    return None

def set_stream(stream):
    if USE_GPU:
        stream.use()


#DATA TRANSFER
def to_gpu(data):
    if USE_GPU:
        return cp.asarray(data)
    return data

def to_cpu(data):
    if USE_GPU:
        return cp.asnumpy(data)
    return data




