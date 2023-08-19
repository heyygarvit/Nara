
import sys
sys.path.append('src')
from advanced.Transformer import create_causal_mask, dropout, layer_norm, pointwise_feed_forward_network, positional_encoding, relu, scaled_dot_product_attention, xavier_init
from core.GPU import cp





def test_xavier_init():
    size = (10, 10)
    result = xavier_init(size)
    assert result.shape == size, f"Expected shape {size}, but got {result.shape}"

test_xavier_init()

# Test for layer_norm
def test_layer_norm():
    x = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)
    result = layer_norm(x)
    assert result.shape == x.shape, f"Expected shape {x.shape}, but got {result.shape}"

test_layer_norm()

# Test for dropout
def test_dropout():
    x = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)
    result = dropout(x, 0.5)
    assert result.shape == x.shape, f"Expected shape {x.shape}, but got {result.shape}"

test_dropout()

# Test for positional_encoding
def test_positional_encoding():
    position = 10
    d_model = 16
    result = positional_encoding(position, d_model)
    assert result.shape == (1, position, d_model), f"Expected shape {(1, position, d_model)}, but got {result.shape}"

test_positional_encoding()

# Test for pointwise_feed_forward_network
def test_pointwise_feed_forward_network():
    d_model = 16
    dff = 32
    w1, b1, w2, b2 = pointwise_feed_forward_network(d_model, dff)
    assert w1.shape == (d_model, dff), f"Expected shape {(d_model, dff)}, but got {w1.shape}"

test_pointwise_feed_forward_network()

# Test for relu
def test_relu():
    x = cp.array([-1, 0, 1], dtype=cp.float32)
    result = relu(x)
    assert cp.array_equal(result, cp.array([0, 0, 1], dtype=cp.float32)), f"Expected [0, 0, 1], but got {result}"

test_relu()

# Test for create_causal_mask
def test_create_causal_mask():
    size = 3
    result = create_causal_mask(size)
    expected = cp.array([[0, -1e9, -1e9], [0, 0, -1e9], [0, 0, 0]])
    assert cp.array_equal(result, expected), f"Expected {expected}, but got {result}"

test_create_causal_mask()

# Test for scaled_dot_product_attention
def test_scaled_dot_product_attention():
    query = cp.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=cp.float32)
    key = query
    value = query
    output, weights = scaled_dot_product_attention(query, key, value)
    assert output.shape == query.shape, f"Expected shape {query.shape}, but got {output.shape}"

test_scaled_dot_product_attention()

# Note: Testing classes like MultiHeadAttention, TransformerBlock, and GPT would require more extensive test cases 
# involving mock data and potentially mock methods. The above tests are basic sanity checks for the functions.

print("All tests passed!")