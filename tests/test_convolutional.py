import numpy as np
import sys
sys.path.append('src')
from layers.Convolutional import Conv2D
from core.GPU import cp 

def test_initialization():
    conv = Conv2D(3, 16, 3)
    assert conv.filters.shape == (16, 3, 3, 3)
    assert conv.biases.shape == (16, 1, 1, 1)
    print("Initialization Test Passed!")

def test_forward():
    conv = Conv2D(3, 16, 3)
    x = cp.random.randn(10, 3, 32, 32)
    out = conv.forward(x)
    assert out.shape == (10, 16, 32, 32)
    print("Forward Pass Test Passed!")

def test_backward():
    conv = Conv2D(3, 16, 3)
    x = cp.random.randn(10, 3, 32, 32)
    dout = cp.random.randn(10, 16, 32, 32)
    out = conv.forward(x)
    dinput = conv.backward(dout)
    assert dinput.shape == x.shape
    print("Backward Pass Test Passed!")

def test_batchnorm():
    conv = Conv2D(3, 16, 3)
    x = cp.random.randn(10, 3, 32, 32)
    out = conv.forward(x)
    assert out.shape == (10, 16, 32, 32)
    print("Batch Normalization Test Passed!")

def test_activations():
    activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']
    for act in activations:
        conv = Conv2D(3, 16, 3, activation=act)
        x = cp.random.randn(10, 3, 32, 32)
        out = conv.forward(x)
        assert out.shape == (10, 16, 32, 32)
    print("Activation Functions Test Passed!")

def test_padding():
    paddings = ['SAME', 'VALID']
    for pad in paddings:
        conv = Conv2D(3, 16, 3, padding=pad)
        x = cp.random.randn(10, 3, 32, 32)
        out = conv.forward(x)
        if pad == 'SAME':
            assert out.shape == (10, 16, 32, 32)
        else:
            assert out.shape == (10, 16, 30, 30)
    print("Padding Test Passed!")

if __name__ == "__main__":
    test_initialization()
    test_forward()
    test_backward()
    test_batchnorm()
    test_activations()
    test_padding()

