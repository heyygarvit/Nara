import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None


    @staticmethod
    def _ensure_tensor(tensor):
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)
        return tensor

    def __repr__(self):
        grad_fn_str = f", grad_fn={self._grad_fn}" if self.requires_grad else ""
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_fn_str})"
    
    #ZERO GRAD

    
    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def zero_(self):
        self.data = 0.0
        return self
    
    def clear_grad(self):
        self.grad = None

   
    
      # BASIC OPERATIONS

    def __add__(self, other):
        other = self._ensure_tensor(other)
        result = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        if result.requires_grad:
            # result._grad_fn = ((self, other), (lambda t, g: g, lambda t, g: g))
            result._grad_fn = (self, other), (lambda t, g: Tensor(g.data), lambda t, g: Tensor(g.data))

        return result
    
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        result = Tensor(self.data - other.data, self.requires_grad or other.requires_grad)
        if result.requires_grad:
            # result._grad_fn = ((self, other), (lambda t, g: g, lambda t, g: -g)) # Gradient functions for subtraction
            # result._grad_fn = (self, other), (lambda t, g: Tensor(g), lambda t, g: Tensor(-g))
            result._grad_fn = (self, other), (lambda t, g: Tensor(g.data), lambda t, g: Tensor(-g.data))


        return result


    def __mul__(self, other):
        other = self._ensure_tensor(other)
        result = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
        if result.requires_grad:
           
            result._grad_fn = (self, other), (lambda t, g: Tensor(g.data * other.data), lambda t, g: Tensor(g.data * self.data))
           
            # result._grad_fn = ((self, other), (lambda t, g: g * other.data, lambda t, g: g * self.data))  # Gradient functions for multiplication
        return result
    
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        if np.any(other.data == 0):
            raise ValueError("Division by zero encountered!!")
        result = Tensor(self.data / other.data, self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result._grad_fn = (self, other), (lambda t, g: Tensor(g.data / other.data), lambda t, g: Tensor(-g.data * self.data / (other.data ** 2)))

            # result._grad_fn = ((self, other), (lambda t, g: g / other.data, lambda t, g: -g * self.data / (other.data ** 2)))  # Gradient functions for division
        return result
    
    def __pow__(self, power):
        result = Tensor(self.data ** power, self.requires_grad)
        if result.requires_grad:
            # result._grad_fn = ((self,), (lambda t, g: power * (t.data ** (power - 1)) * g,))
            #  result._grad_fn = (self,), (lambda t, g: Tensor(power * (t.data ** (power - 1)) * g),)
             result._grad_fn = (self,), (lambda t, g: Tensor(power * (t.data ** (power - 1)) * g.data),)


        return result

    
    # def matmul(self, other):
    #     other = self._ensure_tensor(other)
    #     if self.data.shape[1] != other.data.shape[0]:
    #         raise ValueError(f"Matrix shapes {self.data.shape} and {other.data.shape} are not aligned for multiplication!")
    #     result = Tensor(np.dot(self.data, other.data), self.requires_grad or other.requires_grad)
    #     if result.requires_grad:
    #         result._grad_fn = (self, other), (lambda t, g: g @ other.data.T, lambda t, g: t.data.T @ g)
    #     return result

    # def matmul(self, other):
    #     other = self._ensure_tensor(other)
    #     if self.data.shape[1] != other.data.shape[0]:
    #         raise ValueError(f"Matrix shapes {self.data.shape} and {other.data.shape} are not aligned for multiplication!")
    #     result = Tensor(np.dot(self.data, other.data), self.requires_grad or other.requires_grad)
    #     if result.requires_grad:
    #         result._grad_fn = (self, other), (lambda t, g: Tensor(g.data @ other.data.T), lambda t, g: Tensor(t.data @ g.T if t.requires_grad and t.data.shape[0] > 1 else np.zeros_like(t.data)))

    #         # result._grad_fn = ((self, other), (lambda t, g: g @ other.data.T, lambda t, g: t.data @ g.T if t.requires_grad and t.data.shape[0] > 1 else np.zeros_like(t.data)))
    #     return result
    
    # def matmul(self, other):
    #     other = self._ensure_tensor(other)
    #     if self.data.shape[1] != other.data.shape[0]:
    #         raise ValueError(f"Matrix shapes {self.data.shape} and {other.data.shape} are not aligned for multiplication!")
    #     result = Tensor(np.dot(self.data, other.data), self.requires_grad or other.requires_grad)
    #     if result.requires_grad:
    #         result._grad_fn = (self, other), (lambda t, g: Tensor(g.data @ other.data.T), lambda t, g: Tensor(t.data @ g.data.T if t.requires_grad and t.data.shape[0] > 1 else np.zeros_like(t.data)))
        # return result

    def matmul(self, other):
        other = self._ensure_tensor(other)
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(f"Matrix shapes {self.data.shape} and {other.data.shape} are not aligned for multiplication!")
        result = Tensor(np.dot(self.data, other.data), self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result._grad_fn = (self, other), (lambda t, g: Tensor(g.data @ other.data.T), lambda t, g: Tensor(t.data.T @ g.data))
            # result._grad_fn = (self, other, lambda t, g: Tensor(g.data @ other.data.T), lambda t, g: Tensor(t.data.T @ g.data))

            # result._grad_fn = (self, other), (lambda t, g: self.debug_grad_computation(g, other), lambda t, g: Tensor(t.data.T @ g.data))
            # result._grad_fn = (self, other), (lambda t, g: Tensor(g.data @ other.data.T), lambda t, g: Tensor(t.data.T @ g.data))
        return result
    
    # def debug_grad_computation(self, g, other):
    #     print("g.data:", g.data)
    #     print("other.data.T:", other.data.T)
    #     computed_grad = g.data @ other.data.T
    #     print("Computed grad for a:", computed_grad)
    #     return Tensor(computed_grad)

    
    # def square(self):
    #     result = Tensor(self.data ** 2, self.requires_grad)
    #     if result.requires_grad:
    #         result._grad_fn = (self,), (lambda t, g: 2 * t.data * g,)
    #     return result
  
    def square(self):
        result = Tensor(self.data ** 2, self.requires_grad)
        if self.requires_grad:
            # def grad_fn_square(t, g):
            #     grad_value = 2 * t.data * g
            #     return grad_value           
            # result._grad_fn = ((self,), (lambda t, g: 2 * t.data * g,))
            def grad_fn_square(t, g):
                grad_value = 2 * t.data * g
                return Tensor(grad_value)
            result._grad_fn = (self,), (grad_fn_square,)

        return result


    def mean(self):
        result = Tensor(np.mean(self.data), self.requires_grad)
        if result.requires_grad:
            result._grad_fn = (self,), (lambda t, g: Tensor(np.ones_like(t.data) * g),)

            # result._grad_fn = ((self,), (lambda t, g: np.ones_like(t.data) * g,))
        return result

    # BROADCASTING

    def broadcast_to(self, shape):
        result = Tensor(np.broadcast_to(self.data, shape))
        if self.requires_grad:
            result._grad_fn = lambda: [self]
            result.requires_grad = True
        return result
    
    #TENSOR DECOMPOSITION

    def svd(self):
        u,s, vh = np.linalg.svd(self.data)
        return Tensor(u) , Tensor(s), Tensor(vh)

    def qr(self):
        q, r = np.linalg.qr(self.data)
        return Tensor(q) , Tensor(r)
    

    # ADVANCED REDUCTIONS

    def argmax(self, axis= None):
        return Tensor(np.argmax(self.data, axis=axis)) # Return Tensor object
    

    def cumsum(self, axis= None):
        return Tensor(np.cumsum(self.data, axis=axis)) # Return Tensor object
    
    
    #ELEMENT-WISE OPERATIONS

    def exp(self):
        result = Tensor(np.exp(self.data))
        if result.requires_grad:
            result._grad_fn = lambda: [self * result.exp()]
            result.requires_grad = True
        return result
    

    def log(self):
        result = Tensor(np.log(self.data))
        if self.requires_grad:
            result._grad_fn = lambda: [self / result]
            result.requires_grad = True
        return result
    

    #INVERSE OF A MATRIX

    def inverse(self):
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Matrix must be square to compute its inverse.")
       
        result = Tensor(np.linalg.inv(self.data))
        if self.requires_grad:
           result._grad_fn = (self,), (lambda t, g: -t.inverse().matmul(g).matmul(t.inverse())
)

        #    result._grad_fn = (self,), (lambda t, g: -t.inverse().matmul(g).matmul(t.inverse())), # Tuple structure
        #    result._grad_fn = ((self,), (lambda t, g: -t.inverse().matmul(g).matmul(t.inverse()),))
           result.requires_grad = True
        return result
    


    # You can further add other operations as needed.

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        
        # Print the current tensor and its gradient
        # print(f"Tensor: {self.data}, Gradient Passed: {grad.data}")

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        # print(f"Current tensor: {self}, Gradient: {self.grad}")  # Add this print statement
        if self._grad_fn is not None:
            tensors, grad_fns = self._grad_fn
            grads = [fn(tensor, grad) for tensor, fn in zip(tensors, grad_fns)]
            for tensor, specific_grad in zip(tensors, grads):
                tensor.backward(specific_grad)

        



        # def backward(self, grad=None):
        #     if not self.requires_grad:
        #         return

        #     if grad is None and self.grad is None:
        #         grad = np.ones_like(self.data)

        #     if self.grad is None:
        #         self.grad = grad
        #     else:
        #         self.grad += grad

        #     if self._grad_fn:
        #         # Check if _grad_fn is in the tuple format
        #         if isinstance(self._grad_fn, tuple) and len(self._grad_fn) == 4:
        #             tensors, grad_fns = self._grad_fn[0:2], self._grad_fn[2:]
        #             for tensor, grad_fn in zip(tensors, grad_fns):
        #                 if tensor is not None:
        #                     specific_grad = grad_fn(tensor, grad)
        #                     tensor.backward(specific_grad)
        #         # Check if _grad_fn is a single lambda function
        #         elif callable(self._grad_fn):
        #             grad_fn = self._grad_fn
        #             specific_grad = grad_fn(self, grad)
        #             self.backward(specific_grad)
        #         else:
        #             raise ValueError("Unsupported format for _grad_fn.") 

    # def backward(self, grad=None):
    #     if not self.requires_grad:
    #         return
        
    #     if grad is None and self.grad is None:
    #         self.grad = Tensor(np.ones_like(self.data))
    #     elif self.grad is None:
    #         self.grad = grad

    #     if self._grad_fn:
    #         inputs, gradients = self._grad_fn
    #         for input_tensor, gradient_fn in zip(inputs, gradients):
    #             if input_tensor.requires_grad:
    #                 input_tensor.grad = gradient_fn(self, self.grad)
    #                 input_tensor.backward()

    # def backward(self, grad=None):
    #         if grad is None and self.grad is None:
    #             # Seed gradient
    #             grad = 1.0

    #         if self.grad is None:
    #             self.grad = grad
    #         else:
    #             self.grad += grad

    #         if self._grad_fn is not None:
               
    #             *tensors, grad_fns = self._grad_fn

    #             for tensor in tensors:
    #                 grads = [fn(tensor, grad) for tensor, fn in zip(tensors, grad_fns)]
    #             # Adjust for cases where there's only one grad_fn and no associated tensor
    #             if callable(tensors):  # If 'tensors' is actually our gradient function
    #                 tensors = (self,)
    #                 grad_fns = (grad_fns,)

    #             # Compute the gradients for the precursor tensors.
    #             # grads = [fn(t, grad) for fn in grad_fns]


    #             if not isinstance(tensors, tuple):
    #                 tensors = (tensors,)
    #             grads = [fn(tensor, grad) for tensor, fn in zip(tensors, grad_fns)]
                
    #             # Recursive call: propagate the gradients to the precursor tensors.
    #             for tensor, specific_grad in zip(tensors, grads):
    #                 tensor.backward(specific_grad)

# import numpy as np
# # from tensor import Tensor

# a = Tensor(np.array([[1,2],[3,4]]), requires_grad=True)
# b = Tensor(np.array([[2],[3]]), requires_grad=True)

# # Only perform a subset of operations
# z = a.matmul(b)
# z = z.square()

# # Check the backward propagation
# z.backward()

# print(a.grad)
# print(b.grad)
