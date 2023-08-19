
import sys

import numpy as np
sys.path.append('src')
from core.Tensor import Tensor
from layers.Dense import BatchNorm, Dropout, Layer, Dense, LayerNorm, ReLU, Sigmoid, Softmax, Tanh, constrain_weights, optimized_matmul
from core.GPU import USE_GPU, check_memory


def test_layer():
    layer = Layer()
    assert len(layer.get_parameters()) == 0, "Layer parameters should be empty initially."

def test_dense():
    dense = Dense(5, 3)
    x = Tensor(np.array([[1, 2, 3, 4, 5]]))
    output = dense.forward(x)
    # assert output.shape == (1, 3), "Dense forward pass shape mismatch."
    assert output.data.shape == (1, 3), "Dense forward pass shape mismatch." 

    assert len(dense.get_parameters()) == 2, "Dense should have 2 parameters (weights and bias)."

def test_sigmoid():
    sigmoid = Sigmoid()
    x = np.array([-1, 0, 1])
    output = sigmoid(x)
    assert np.allclose(output, [0.26894142, 0.5, 0.73105858]), "Sigmoid output mismatch."

def test_tanh():
    tanh = Tanh()
    x = np.array([-1, 0, 1])
    output = tanh(x)
    assert np.allclose(output, np.tanh(x)), "Tanh output mismatch."

def test_relu():
    relu = ReLU()
    x = np.array([-1, 0, 1])
    output = relu(x)
    assert np.allclose(output, [0, 0, 1]), "ReLU output mismatch."

def test_softmax():
    softmax = Softmax()
    x = np.array([1, 2, 3])
    output = softmax(x)
    expected_output = [0.09003057, 0.24472847, 0.66524096]
    assert np.allclose(output, expected_output), "Softmax output mismatch."

def test_batchnorm():
    bn = BatchNorm(3)
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    output = bn.forward(x)
    assert output.data.shape == (2, 3), "BatchNorm forward pass shape mismatch."

def test_dropout():
    dropout = Dropout(0.5)
    x = Tensor(np.array([1, 2, 3]))
    dropout.train()
    output_train = dropout.forward(x)
    dropout.eval()
    output_eval = dropout.forward(x)
    assert np.allclose(output_eval.data, x.data), "Dropout in eval mode should not change input."

def test_layernorm():
    ln = LayerNorm(3)
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    output = ln.forward(x)
    assert output.data.shape == (2, 3), "LayerNorm forward pass shape mismatch."

def test_check_memory():
    # This test will just run the function to ensure no exceptions are raised.
    # Actual memory checks will depend on the system's available memory.
    check_memory(1000)

# def test_optimized_matmul():
#     A = np.array([[1, 2], [3, 4]])
#     B = np.array([[2, 3], [4, 5]])
#     output = optimized_matmul(A, B)
#     expected_output = np.array([[10, 13], [22, 29]])
#     assert np.allclose(output, expected_output), "Optimized matmul output mismatch."

def test_constrain_weights():
    weights = np.array([-2, -1, 0, 1, 2])
    output = constrain_weights(weights)
    expected_output = np.array([-1, -1, 0, 1, 1])
    assert np.allclose(output, expected_output), "Constrain weights output mismatch."

if __name__ == "__main__":
    test_layer()
    test_dense()
    test_sigmoid()
    test_tanh()
    test_relu()
    test_softmax()
    test_batchnorm()
    test_dropout()
    test_layernorm()
    test_check_memory()
    # test_optimized_matmul()
    test_constrain_weights()
    print("All tests passed!")