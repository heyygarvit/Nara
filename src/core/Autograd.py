import numpy as np
import math

class Variable:
    def __init__(self, data, requires_grad = False, _children=(), _op="" , label=""):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self._grad_fn = None
        self._backward = lambda:[]
        self._prev = set(_children)
        self._op = _op
        self.label = label


    def zero_grad(self):
        self.grad = 0.0

    def __repr__(self):
        return f"Variable(data={self.data})"

    # BASIC OPERATIONS
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, requires_grad= True, _children= (self,other), _op="+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, requires_grad=True, _children=(self, other), _op='*')
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        base = float(self.data)
        exponent = float(other.data) if isinstance(other, Variable) else float(other)
        out = Variable(base ** exponent, requires_grad=True, _children=(self,), _op=f'**{exponent}')
        def _backward():
            self.grad += exponent * (base ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    
    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data / other.data, requires_grad=True, _children=(self, other), _op='/')
        
        def _backward():
            # GRADIENT WITH RESPECT TO THE NUMERATOR
            self.grad += 1.0 / other.data * out.grad
            # GRADIENT WITH RESPECT TO THE DENOMINATOR
            other.grad += -self.data / (other.data**2) * out.grad
        out._backward = _backward
        return out



    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data - other.data, requires_grad=True, _children=(self, other), _op='-')
        def _backward():
            if self.requires_grad:
                self.grad += 1.0 * out.grad
            if other.requires_grad:
                other.grad += -1.0 * out.grad
        out._backward = _backward
        return out


    def __neg__(self):
        return self * -1
    
    
    # ACTIVATION FUNCTIONS
    
    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Variable(t, requires_grad=True, _children=(self,), _op='tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out


    def relu(self):
        out = Variable(0 if self.data < 0 else self.data, requires_grad=True, _children=(self,), _op='ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        result = 1.0 / (1.0 + np.exp(-self.data))
        out = Variable(result, requires_grad=True, _children=(self,), _op="sigmoid")
        def _backward():
            s = out.data
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    def softmax(self):
        exp_data = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))  # FOR NUMERICAL STABILITY
        result = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        out = Variable(result, requires_grad=True, _children=(self,), _op="softmax")
        def _backward():
            s = out.data
            # GENERAL GRADIENT COMPUTATION FOR SOFTMAX
            for i in range(len(s)):
                jacobian = -np.outer(s[i], s[i]) + np.diag(s[i])
                self.grad[i] = jacobian @ out.grad[i]
        out._backward = _backward

        return out

    def softmax_cross_entropy(self, target):
        # SOFTMAX
        exp_data = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        softmax_result = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        # CROSS ENTROPY
        if len(softmax_result.shape) == 1:
            log_likelihood = -np.log(softmax_result[target])
        else:
            log_likelihood = -np.log(softmax_result[range(target.shape[0]), target])

        loss = np.sum(log_likelihood) / target.shape[0]
        out = Variable(loss, requires_grad=True, _children=(self,), _op="softmax_cross_entropy")
        def _backward():
            grad = softmax_result
            grad[range(target.shape[0]), target] -= 1
            grad /= target.shape[0]
            self.grad += grad
        out._backward = _backward
        return out
    


    def exp(self):
        out = Variable(math.exp(self.data), requires_grad=True, _children=(self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out