import sys
sys.path.append('src')
from core.GPU import cp

# Xavier Initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / cp.sqrt(in_dim / 2.)
    return cp.random.randn(*size) * xavier_stddev

# Layer Normalization
def layer_norm(x, epsilon=1e-6):
    mean = cp.mean(x, axis=-1, keepdims=True)
    std = cp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

# Dropout (for demonstration, in practice, use a library function)
def dropout(x, rate, training=True, seed=None):
    """
    Apply dropout to the input tensor.

    Parameters:
    - x: Input tensor.
    - rate: Dropout rate, a float between 0 and 1.
    - training: Whether the model is in training mode. If False, no dropout is applied.
    - seed: Optional random seed for reproducibility.

    Returns:
    - Tensor with dropout applied.

    """

    if not (0 <= rate < 1):
        raise ValueError(f"Invalid dropout rate: {rate}. It should be between 0 and 1.")
    
    if not training:
        return x

    if seed is not None:
        cp.random.seed(seed)

    mask = cp.random.rand(*x.shape) > rate
    return x * mask / (1.0 - rate)


# Positional Encoding
def positional_encoding(position, d_model):
    angle_rads = cp.arange(position)[:, cp.newaxis] / cp.power(10000, (2 * (cp.arange(d_model)[cp.newaxis, :] // 2)) / cp.float32(d_model))
    angle_rads[:, 0::2] = cp.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = cp.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[cp.newaxis, ...]
    return pos_encoding

# Adding bias terms and activation function in Feed Forward Network
def pointwise_feed_forward_network(d_model, dff):
    w1 = xavier_init((d_model, dff))
    b1 = cp.zeros((dff,))
    w2 = xavier_init((dff, d_model))
    b2 = cp.zeros((d_model,))
    return w1, b1, w2, b2

def relu(x):
    return cp.maximum(0, x)

def softmax(x):
    e_x = cp.exp(x - cp.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)




# Modify the TransformerBlock to use the biases and activation function
class TransformerBlock:
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) should be divisible by num_heads ({num_heads}).")
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn_w1, self.ffn_b1, self.ffn_w2, self.ffn_b2 = pointwise_feed_forward_network(d_model, dff)
        self.dropout_rate = rate

    def forward(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = dropout(attn_output, self.dropout_rate, training)
        out1 = layer_norm(x + attn_output)
        ffn_output = relu(cp.matmul(out1, self.ffn_w1) + self.ffn_b1)
        ffn_output = dropout(ffn_output, self.dropout_rate, training)
        ffn_output = cp.matmul(ffn_output, self.ffn_w2) + self.ffn_b2
        out2 = layer_norm(out1 + ffn_output)
        return out2

# Masking
def create_causal_mask(size):
    mask = 1 - cp.tril(cp.ones((size, size)))
    return mask * -1e9

# Scaled Dot Product Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = cp.matmul(query, key.T)  # Transpose key here
    d_k = key.shape[-1]
    scaled_attention_logits = matmul_qk / cp.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += mask

   
    attention_weights = softmax(scaled_attention_logits)
    output = cp.matmul(attention_weights, value)

    return output, attention_weights

# GPT Model
# Modify the GPT's forward method to use the training flag
class GPT:
    def __init__(self, d_model, num_heads, dff, num_layers, vocab_size, max_position_encoding, rate=0.1):
        self.embedding = xavier_init((vocab_size, d_model))
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.final_layer = xavier_init((d_model, vocab_size))

    def forward(self, x, training=True):
        seq_len = x.shape[1]
        x = cp.matmul(x, self.embedding)
        x += self.pos_encoding[:, :seq_len, :]

        mask = create_causal_mask(seq_len)
        for block in self.transformer_blocks:
            x = block.forward(x, training, mask)

        logits = cp.matmul(x, self.final_layer)
        return logits

# Loss Computation
def compute_loss(logits, true_next_words):
    probabilities = cp.softmax(logits, axis=-1)
    loss = -cp.sum(true_next_words * cp.log(probabilities + 1e-9))
    return loss

# MultiHead Attention
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) should be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = xavier_init((d_model, d_model))
        self.wk = xavier_init((d_model, d_model))
        self.wv = xavier_init((d_model, d_model))
        self.dense = xavier_init((d_model, d_model))
        self.bq = cp.zeros((d_model,))
        self.bk = cp.zeros((d_model,))
        self.bv = cp.zeros((d_model,))
        self.dense_bias = cp.zeros((d_model,))

    def split_heads(self, x, batch_size):
        x = cp.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return cp.transpose(x, (0, 2, 1, 3))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = cp.matmul(query, self.wq) + self.bq
        key = cp.matmul(key, self.wk) + self.bk
        value = cp.matmul(value, self.wv) + self.bv

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention_output, _ = scaled_dot_product_attention(query, key, value, mask)
        attention_output = cp.transpose(attention_output, (0, 2, 1, 3))
        concat_attention = cp.reshape(attention_output, (batch_size, -1, self.d_model))
        output = cp.matmul(concat_attention, self.dense) + self.dense_bias

        return output
    

