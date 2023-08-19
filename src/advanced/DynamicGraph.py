import logging
import sys
sys.path.append('src')
from core.GPU import cp, USE_GPU

class Node:
    def __init__(self, value, grad=None):
        # Ensure the value and gradient are on the correct device (GPU/CPU)
        self.value = cp.array(value)
        self.grad = cp.array(grad if grad is not None else 0)
        self.backward_fn = None

    def backward(self, grad=None):
        if grad is not None:
            self.grad = cp.array(grad)

        if self.backward_fn is not None:
            self.backward_fn(self.grad)

    def __add__(self, other):
        if isinstance(other, (int, float)):  # Scalar addition
            result = Node(self.value + other)
            def backward(grad):
                self.grad += grad
            result.backward_fn = backward
        else:  # Node addition
            result = Node(self.value + other.value)
            def backward(grad):
                self.grad += grad
                other.grad += grad
            result.backward_fn = backward
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Scalar multiplication
            result = Node(self.value * other)
            def backward(grad):
                self.grad += other * grad
            result.backward_fn = backward
        else:  # Node multiplication
            result = Node(self.value * other.value)
            def backward(grad):
                self.grad += other.value * grad
                other.grad += self.value * grad
            result.backward_fn = backward
        return result

    def __sub__(self, other):
        if isinstance(other, (int, float)):  # Scalar subtraction
            result = Node(self.value - other)
            def backward(grad):
                self.grad += grad
            result.backward_fn = backward
        else:  # Node subtraction
            result = Node(self.value - other.value)
            def backward(grad):
                self.grad += grad
                other.grad -= grad
            result.backward_fn = backward
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):  # Scalar division
            if other == 0:
                raise ValueError("Division by zero!")
            result = Node(self.value / other)
            def backward(grad):
                self.grad += grad / other
            result.backward_fn = backward
        else:  # Node division
            if other.value == 0:
                raise ValueError("Division by zero!")
            result = Node(self.value / other.value)
            def backward(grad):
                self.grad += grad / other.value
                other.grad -= (self.value * grad) / (other.value ** 2)
            result.backward_fn = backward
        return result

    # Implementing in-place operations
    def __iadd__(self, other):
        self.value += other.value
        def backward(grad):
            self.grad += grad
            other.grad += grad
        self.backward_fn = backward
        return self

    def __isub__(self, other):
        self.value -= other.value
        def backward(grad):
            self.grad += grad
            other.grad -= grad
        self.backward_fn = backward
        return self

    def __imul__(self, other):
        self.value *= other.value
        def backward(grad):
            self.grad += other.value * grad
            other.grad += self.value * grad
        self.backward_fn = backward
        return self

    def __idiv__(self, other):
        if other.value == 0:
            raise ValueError("Division by zero!")
        self.value /= other.value
        def backward(grad):
            self.grad += grad / other.value
            other.grad -= (self.value * grad) / (other.value ** 2)
        self.backward_fn = backward
        return self

class DynamicGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, value, grad=None):
        node = Node(value, grad)
        self.nodes.append(node)
        return node

    def backward(self, start_node):
        start_node.backward()

    def zero_grad(self):
        for node in self.nodes:
            node.grad = cp.array(0)