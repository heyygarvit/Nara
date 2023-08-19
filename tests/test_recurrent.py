import numpy as np
import sys
sys.path.append('src')
from layers.Recurrent import RNN
from core.GPU import cp
from utils.EvalutionMetrics import Metrics



def test_RNN():
    # Initialize parameters
    input_size = 5
    hidden_size = 10
    output_size = 1
    learning_rate = 0.01
    clip_value = 5
    dropout_rate = 0.5
    bidirectional = False
    activation = 'tanh'
    batch_norm = False

    # Create RNN instance
    rnn = RNN(input_size, hidden_size, output_size, learning_rate, clip_value, dropout_rate, bidirectional, activation, batch_norm)

    # Create dummy data
    x = cp.random.randn(10, input_size)
    y_true = cp.random.randn(10, output_size)
    h_prev = cp.zeros((10, hidden_size))

    # Test forward pass
    y_pred, h_next = rnn.forward(x, h_prev)
    assert y_pred.shape == (10, output_size)
    assert h_next.shape == (10, hidden_size)

    # Test backward pass
    dY = y_pred - y_true
    dWx, dWh, dbh, dWy, dby = rnn.backward(dY, h_next, h_prev, x)

    # Test training function
    loss, h_next = rnn.train(x, y_true, h_prev)
    assert isinstance(loss, cp.ndarray)
    assert h_next.shape == (10, hidden_size)

    # Test prediction function
    y_pred = rnn.predict(x, h_prev)
    assert y_pred.shape == (10, output_size)

    # Test evaluation
    metrics = rnn.evaluate(x, y_true, h_prev)
    assert isinstance(metrics, Metrics)

    print("All tests passed!")

test_RNN()
