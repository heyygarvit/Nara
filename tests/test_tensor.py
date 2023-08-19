import numpy as np
import sys
sys.path.append('src')
from core.Tensor import Tensor



def test_tensor_basic_operations():
    # __add__
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a + b
    assert np.array_equal(c.data, [5, 7, 9])
    c.backward(Tensor([1, 1, 1]))
    assert np.array_equal(a.grad.data, [1, 1, 1])
    assert np.array_equal(b.grad.data, [1, 1, 1])
    
    # __sub__
    d = a - b
    assert np.array_equal(d.data, [-3, -3, -3])
    d.backward(Tensor([1, 1, 1]))
    assert np.array_equal(a.grad.data, [2, 2, 2])  # 1 + 1
    assert np.array_equal(b.grad.data, [0, 0, 0])  # 1 - 1
    
    # __mul__
    e = a * b
    assert np.array_equal(e.data, [4, 10, 18])
    e.backward(Tensor([1, 1, 1]))
    print(a.grad.data)
    assert np.array_equal(a.grad.data, [4, 5, 6])
    assert np.array_equal(b.grad.data, [1, 2, 3])

    # assert np.array_equal(a.grad.data, [5, 7, 9])  # 2 + 3, 2 + 5, 2 + 6
    # assert np.array_equal(b.grad.data, [2, 4, 6])  # 1 + 1, 2 + 2, 3 + 3
    
    # ... Add more operations as needed

def test_tensor_advanced_operations():
    # matmul
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[2, 0], [1, 3]], requires_grad=True)
    c = a.matmul(b)
    assert np.array_equal(c.data, [[4, 6], [10, 12]])
    c.backward(Tensor([[1, 1], [1, 1]]))
    assert np.array_equal(a.grad.data, [[3, 3], [3, 3]])
    assert np.array_equal(b.grad.data, [[4, 4], [6, 6]])
    
    # ... Add more operations as needed

# Run the tests
test_tensor_basic_operations()
test_tensor_advanced_operations()
print("All tests passed!")






# def test_tensor_basic_operations():
#     # Addition
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([4, 5, 6], requires_grad=True)
#     c = a + b
#     c.backward(Tensor([1, 1, 1]))
#     print("Gradients after addition:")
#     print("a.grad:", a.grad.data)
#     print("b.grad:", b.grad.data)
#     a.zero_grad()  # Zero out the gradients
#     b.zero_grad()
    
#     # Subtraction
#     d = a - b
#     d.backward(Tensor([1, 1, 1]))
#     print("\nGradients after subtraction:")
#     print("a.grad:", a.grad.data)
#     print("b.grad:", b.grad.data)
#     a.zero_grad()  # Zero out the gradients
#     b.zero_grad()
    
#     # Multiplication
#     e = a * b
#     e.backward(Tensor([1, 1, 1]))
#     print("\nGradients after multiplication:")
#     print("a.grad:", a.grad.data)
#     print("b.grad:", b.grad.data)
#     a.zero_grad()
#     b.zero_grad()
#     # ... Continue for other operations as needed



# test_tensor_basic_operations()


# Tensor: [5 7 9], Gradient Passed: [1 1 1]
# Tensor: [1 2 3], Gradient Passed: [1 1 1]
# Tensor: [4 5 6], Gradient Passed: [1 1 1]
# # Gradients after addition:
# a.grad: [1 1 1]
# b.grad: [1 1 1]
# Tensor: [-3 -3 -3], Gradient Passed: [1 1 1]
# Tensor: [1 2 3], Gradient Passed: [1 1 1]
# Tensor: [4 5 6], Gradient Passed: [-1 -1 -1]

# # Gradients after subtraction:
# a.grad: [1 1 1]
# b.grad: [-1 -1 -1]
# Tensor: [ 4 10 18], Gradient Passed: [1 1 1]
# Tensor: [1 2 3], Gradient Passed: [4 5 6]
# Tensor: [4 5 6], Gradient Passed: [1 2 3]

# # Gradients after multiplication:
# a.grad: [4 5 6]
# b.grad: [1 2 3]







# def test_tensor_initialization():
#     t = Tensor([1, 2, 3])
#     assert np.array_equal(t.data, [1, 2, 3])
#     assert t.requires_grad == False
#     assert t.grad == None
#     assert t._grad_fn == None

# def test_tensor_addition():
#     t1 = Tensor([1, 2, 3], requires_grad=True)
#     t2 = Tensor([4, 5, 6], requires_grad=True)
#     t3 = t1 + t2
#     assert np.array_equal(t3.data, [5, 7, 9])
#     assert t3.requires_grad == True

# def test_tensor_subtraction():
#     t1 = Tensor([1, 2, 3], requires_grad=True)
#     t2 = Tensor([4, 5, 6], requires_grad=True)
#     t3 = t1 - t2
#     assert np.array_equal(t3.data, [-3, -3, -3])
#     assert t3.requires_grad == True

# def test_tensor_multiplication():
#     t1 = Tensor([1, 2, 3], requires_grad=True)
#     t2 = Tensor([4, 5, 6], requires_grad=True)
#     t3 = t1 * t2
#     assert np.array_equal(t3.data, [4, 10, 18])
#     assert t3.requires_grad == True

# def test_tensor_division():
#     t1 = Tensor([1, 2, 3], requires_grad=True)
#     t2 = Tensor([4, 5, 6], requires_grad=True)
#     t3 = t1 / t2
#     assert np.allclose(t3.data, [0.25, 0.4, 0.5])
#     assert t3.requires_grad == True

# def test_tensor_matmul():
#     t1 = Tensor([[1, 2], [3, 4]], requires_grad=True)
#     t2 = Tensor([[2, 0], [1, 3]], requires_grad=True)
#     t3 = t1.matmul(t2)
#     assert np.array_equal(t3.data, [[4, 6], [10, 12]])
#     assert t3.requires_grad == True

# def test_tensor_broadcast_to():
#     t = Tensor([1, 2, 3], requires_grad=True)
#     t_broad = t.broadcast_to((3, 3))
#     assert np.array_equal(t_broad.data, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])
#     assert t_broad.requires_grad == True

# def test_tensor_svd():
#     t = Tensor([[4, 0], [0, 3]], requires_grad=True)
#     u, s, vh = t.svd()
#     assert np.array_equal(u.data, np.eye(2))
#     assert np.array_equal(s.data, [4, 3])
#     assert np.array_equal(vh.data, np.eye(2))

# def test_tensor_qr():
#     t = Tensor([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], requires_grad=True)
#     q, r = t.qr()
#     # Add assertions based on expected QR decomposition

# def test_tensor_argmax():
#     t = Tensor([1, 5, 3], requires_grad=True)
#     assert t.argmax().data == 1

# def test_tensor_cumsum():
#     t = Tensor([1, 2, 3], requires_grad=True)
#     assert np.array_equal(t.cumsum().data, [1, 3, 6])

# def test_tensor_exp():
#     t = Tensor([1, 2, 3], requires_grad=True)
#     assert np.allclose(t.exp().data, np.exp([1, 2, 3]))

# def test_tensor_log():
#     t = Tensor([1, 2, 3], requires_grad=True)
#     assert np.allclose(t.log().data, np.log([1, 2, 3]))

# def test_tensor_inverse():
#     t = Tensor([[4, 7], [2, 6]], requires_grad=True)
#     inv_t = t.inverse()
#     expected_inv = np.linalg.inv([[4, 7], [2, 6]])
#     assert np.allclose(inv_t.data, expected_inv)

# def test_tensor_backward():
#     t1 = Tensor([1, 2, 3], requires_grad=True)
#     t2 = Tensor([4, 5, 6], requires_grad=True)
#     t3 = t1 + t2
#     t3.backward(np.array([1, 1, 1]))
#     print("t1.grad",t1.grad)
#     print("t2.grad",t2.grad)
#     assert np.array_equal(t1.grad, [1, 1, 1])
   
#     assert np.array_equal(t2.grad, [1, 1, 1])

# # Run tests
# test_tensor_initialization()
# test_tensor_addition()
# test_tensor_subtraction()
# test_tensor_multiplication()
# test_tensor_division()
# test_tensor_matmul()
# test_tensor_broadcast_to()
# test_tensor_svd()
# test_tensor_qr()
# test_tensor_argmax()
# test_tensor_cumsum()
# test_tensor_exp()
# test_tensor_log()
# test_tensor_inverse()
# test_tensor_backward()

# print("All tests passed!")



# def test_tensor_addition():
#     # Create two tensors
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([4, 5, 6], requires_grad=True)
    
#     # Perform addition
#     c = a + b
#     assert c.data.tolist() == [5, 7, 9]
    
#     # Backward pass
#     c.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert a.grad.data.tolist() == [1, 1, 1]
#     assert b.grad.data.tolist() == [1, 1, 1]

# def test_tensor_subtraction():
#     # Create two tensors
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([4, 5, 6], requires_grad=True)
    
#     # Perform subtraction
#     c = a - b
#     assert c.data.tolist() == [-3, -3, -3]
    
#     # Backward pass
#     c.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert a.grad.data.tolist() == [1, 1, 1]
#     assert b.grad.data.tolist() == [-1, -1, -1]

# def test_tensor_multiplication():
#     # Create two tensors
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([4, 5, 6], requires_grad=True)
    
#     # Perform multiplication
#     c = a * b
#     assert c.data.tolist() == [4, 10, 18]
    
#     # Backward pass
#     c.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert a.grad.data.tolist() == [4, 5, 6]
#     assert b.grad.data.tolist() == [1, 2, 3]

# def test_tensor_division():
#     # Create two tensors
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([4, 5, 6], requires_grad=True)
    
#     # Perform division
#     c = a / b
#     assert c.data.tolist() == [0.25, 0.4, 0.5]
    
#     # Backward pass
#     c.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert a.grad.data.tolist() == [0.25, 0.2, 0.16666666666666666]
#     assert b.grad.data.tolist() == [-0.0625, -0.08, -0.08333333333333333]




# def test_tensor_matmul():
#     # Create two tensors
#     a = Tensor([[1, 2], [3, 4]], requires_grad=True)
#     b = Tensor([[2, 0], [1, 3]], requires_grad=True)
    
#     # Perform matrix multiplication
#     c = a.matmul(b)
#     assert c.data.tolist() == [[4, 6], [10, 12]]
    
#     # Backward pass
#     c.backward(Tensor([[1, 1], [1, 1]]))
    
#     # Check gradients (This is a simplified check, actual gradients can be more complex)
#     # assert np.array_equal(a.grad.data, [[3, 3], [3, 3]])
#     print("a.grad.data", a.grad.data)
#     assert np.array_equal(a.grad.data, [[2, 4], [2, 4]])

#     print("b.grad.data", b.grad.data)
#     assert np.array_equal(b.grad.data, [[4, 4], [6, 6]])

# def test_tensor_square():
#     # Create a tensor
#     a = Tensor([1, 2, 3], requires_grad=True)
    
#     # Square the tensor
#     b = a.square()
#     assert b.data.tolist() == [1, 4, 9]
    
#     # Backward pass
#     b.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert a.grad.data.tolist() == [2, 4, 6]

# def test_tensor_mean():
#     # Create a tensor
#     a = Tensor([1, 2, 3, 4], requires_grad=True)
    
#     # Compute the mean
#     b = a.mean()
#     assert b.data == 2.5
    
#     # Backward pass
#     b.backward(Tensor(1))
    
#     # Check gradients
#     assert a.grad.data.tolist() == [0.25, 0.25, 0.25, 0.25]

# def test_tensor_exp():
#     # Create a tensor
#     a = Tensor([1, 2, 3], requires_grad=True)
    
#     # Compute the exponential
#     b = a.exp()
    
#     # Backward pass
#     b.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert np.allclose(a.grad.data, b.data, atol=1e-6)

# def test_tensor_log():
#     # Create a tensor
#     a = Tensor([1, 2, 3], requires_grad=True)
    
#     # Compute the logarithm
#     b = a.log()
    
#     # Backward pass
#     b.backward(Tensor([1, 1, 1]))
    
#     # Check gradients
#     assert np.allclose(a.grad.data, [1, 0.5, 1/3], atol=1e-6)

# def test_tensor_inverse():
#     # Create a tensor
#     a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    
#     # Compute the inverse
#     b = a.inverse()
    
#     # Expected inverse
#     expected_inverse = [[-2, 1], [1.5, -0.5]]
#     assert np.allclose(b.data, expected_inverse, atol=1e-6)
    
#     # Backward pass (This is a bit more complex due to the nature of matrix inversion)
#     b.backward(Tensor([[1, 0], [0, 1]]))
    
#     # Check gradients (This is a simplified check, actual gradients can be more complex)
#     assert np.array_equal(a.grad.data, [[-5.5, 4], [4, -2.5]])

# # Run the tests
# test_tensor_matmul()
# test_tensor_square()
# test_tensor_mean()
# test_tensor_exp()
# test_tensor_log()
# test_tensor_inverse()
# print("All tests passed!")
# # Run the tests
# test_tensor_addition()
# test_tensor_subtraction()
# test_tensor_multiplication()
# test_tensor_division()
# print("All tests passed!")