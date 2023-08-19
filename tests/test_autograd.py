import numpy as np
import sys
sys.path.append('src')
from core.Autograd import Variable


import numpy as np

def test_variable_creation():
    v = Variable(5)
    assert v.data == 5
    assert not v.requires_grad

def test_variable_addition():
    v1 = Variable(5, requires_grad=True)
    v2 = Variable(3, requires_grad=True)
    v3 = v1 + v2
    assert v3.data == 8
    v3.grad = 1.0
    v3._backward()
    assert v1.grad == 1.0
    assert v2.grad == 1.0

def test_variable_power():
    v = Variable(2, requires_grad=True)
    v2 = v ** 3
    assert v2.data == 8
    v2.grad = 1.0
    v2._backward()
    assert v.grad == 12.0

def test_variable_division():
    v1 = Variable(6, requires_grad=True)
    v2 = Variable(3, requires_grad=True)
    v3 = v1 / v2
    assert v3.data == 2
    v3.grad = 1.0
    v3._backward()
    assert v1.grad == 1/3
    assert np.isclose(v2.grad, -2/3)

    


def test_variable_subtraction():
    v1 = Variable(5, requires_grad=True)
    v2 = Variable(3, requires_grad=True)
    v3 = v1 - v2
    assert v3.data == 2
    v3.grad = 1.0
    v3._backward()
    assert v1.grad == 1.0
    print(v2.grad)
    assert v2.grad == -1.0

def test_variable_negation():
    v = Variable(5, requires_grad=True)
    v2 = -v
    assert v2.data == -5
    v2.grad = 1.0
    v2._backward()
    assert v.grad == -1.0

def test_variable_tanh():
    v = Variable(0.5, requires_grad=True)
    v2 = v.tanh()
    assert np.isclose(v2.data, np.tanh(0.5))
    v2.grad = 1.0
    v2._backward()
    assert np.isclose(v.grad, 1 - np.tanh(0.5)**2)

def test_variable_relu():
    v = Variable(-0.5, requires_grad=True)
    v2 = v.relu()
    assert v2.data == 0
    v2.grad = 1.0
    v2._backward()
    assert v.grad == 0

def test_variable_sigmoid():
    v = Variable(0, requires_grad=True)
    v2 = v.sigmoid()
    assert np.isclose(v2.data, 0.5)
    v2.grad = 1.0
    v2._backward()
    assert np.isclose(v.grad, 0.25)

def test_variable_softmax():
    v = Variable([2.0, 1.0, 0.1], requires_grad=True)
    v2 = v.softmax()
    expected = np.exp([2.0, 1.0, 0.1]) / np.sum(np.exp([2.0, 1.0, 0.1]))
    assert np.allclose(v2.data, expected)

def test_variable_softmax_cross_entropy():
    v = Variable([2.0, 1.0, 0.1], requires_grad=True)
    target = np.array([0])
    v2 = v.softmax_cross_entropy(target)
    expected = -np.log(np.exp(2.0) / np.sum(np.exp([2.0, 1.0, 0.1])))
    assert np.isclose(v2.data, expected)

def test_variable_exp():
    v = Variable(2, requires_grad=True)
    v2 = v.exp()
    assert np.isclose(v2.data, np.exp(2))
    v2.grad = 1.0
    v2._backward()
    assert np.isclose(v.grad, np.exp(2))

if __name__ == "__main__":
    test_variable_creation()
    test_variable_addition()
    test_variable_power()
    test_variable_division()
    test_variable_subtraction()
    test_variable_negation()
    test_variable_tanh()
    test_variable_relu()
    test_variable_sigmoid()
    test_variable_softmax()
    test_variable_softmax_cross_entropy()
    test_variable_exp()
    print("All tests passed!")
