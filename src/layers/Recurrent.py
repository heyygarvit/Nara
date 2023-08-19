import numpy as np
import sys
sys.path.append('src')
from core.GPU import cp, USE_GPU, check_memory, convert_dtype
from utils.EvalutionMetrics import Metrics

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, clip_value=5, dropout_rate=0.5, bidirectional=False, activation='tanh', batch_norm=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm

        # Xavier initialization
        self.Wx = convert_dtype(cp.random.randn(input_size, hidden_size) / cp.sqrt(input_size), 'float32')
        self.Wh = convert_dtype(cp.random.randn(hidden_size, hidden_size) / cp.sqrt(hidden_size), 'float32')
        self.bh = convert_dtype(cp.zeros(hidden_size), 'float32')
        self.Wy = convert_dtype(cp.random.randn(hidden_size, output_size) / cp.sqrt(hidden_size), 'float32')
        self.by = convert_dtype(cp.zeros(output_size), 'float32')

        if self.bidirectional:
            self.Wh_reverse = convert_dtype(cp.random.randn(hidden_size, hidden_size) / cp.sqrt(hidden_size), 'float32')
            self.bh_reverse = convert_dtype(cp.zeros(hidden_size), 'float32')

        if self.batch_norm:
            self.gamma = cp.ones((1, hidden_size))
            self.beta = cp.zeros((1, hidden_size))
            self.running_mean = cp.zeros((1, hidden_size))
            self.running_var = cp.zeros((1, hidden_size))
            self.eps = 1e-5
            self.momentum = 0.9

        # Activation function
        if activation == 'tanh':
            self.activation = cp.tanh
            self.activation_name = 'tanh'
        elif activation == 'relu':
            self.activation = lambda x: cp.maximum(0, x)
            self.activation_name = 'relu'
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + cp.exp(-x))
            self.activation_name = 'sigmoid'
        else:
            raise ValueError("Unsupported activation function")


    def forward(self, x, h_prev):
        pre_activation = cp.dot(x, self.Wx) + cp.dot(h_prev, self.Wh) + self.bh
        if self.batch_norm:
            mean = cp.mean(pre_activation, axis=0)
            var = cp.var(pre_activation, axis=0)
            pre_activation = (pre_activation - mean) / cp.sqrt(var + self.eps)
            pre_activation = self.gamma * pre_activation + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        h_next = self.activation(pre_activation)
        if self.dropout_rate > 0:
            self.dropout_mask = (cp.random.rand(*h_next.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            h_next *= self.dropout_mask
        y_pred = cp.dot(h_next, self.Wy) + self.by
        return y_pred, h_next
    

    def set_learning_rate(self, epoch):
        decay_rate = 0.95  # You can tune this value
        decay_step = 10    # You can tune this value
        if epoch % decay_step == 0:
            self.learning_rate *= decay_rate

    def backward(self, dY, h, h_prev, x):
        dWy = cp.dot(h.T, dY)
        dby = cp.sum(dY, axis=0)
        dh = cp.dot(dY, self.Wy.T)

        if self.activation_name == 'tanh':
            dh_raw = (1 - h * h) * dh
        elif self.activation_name == 'relu':
            dh_raw = (h > 0) * dh
        elif self.activation_name == 'sigmoid':
            sigmoid = self.activation(h)
            dh_raw = sigmoid * (1 - sigmoid) * dh

        dWh = cp.dot(h_prev.T, dh_raw)
        dbh = cp.sum(dh_raw, axis=0)
        dWx = cp.dot(x.T, dh_raw)
        return dWx, dWh, dbh, dWy, dby


    def clip_gradients(self, gradients):
        total_norm = cp.sqrt(sum(cp.sum(grad ** 2) for grad in gradients))
        scale = self.clip_value / (total_norm + 1e-6)
        if total_norm > self.clip_value:
            gradients = [grad * scale for grad in gradients]
        return gradients

    def update(self, dWx, dWh, dbh, dWy, dby):
        gradients = [dWx, dWh, dbh, dWy, dby]
        dWx, dWh, dbh, dWy, dby = self.clip_gradients(gradients)
        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.bh -= self.learning_rate * dbh
        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    def train(self, x, y_true, h_prev):
        y_pred, h_next = self.forward(x, h_prev)
        loss_value = Metrics(y_true, y_pred).mean_squared_error()
        loss = cp.array([loss_value])  # Convert scalar to cp.ndarray
        dY = y_pred - y_true
        dWx, dWh, dbh, dWy, dby = self.backward(dY, h_next, h_prev, x)
        self.update(dWx, dWh, dbh, dWy, dby)
        return loss, h_next

    def predict(self, x, h_prev):
        y_pred, _ = self.forward(x, h_prev)
        return y_pred

    def evaluate(self, x, y_true, h_prev):
        y_pred = self.predict(x, h_prev)
        metrics = Metrics(y_true, y_pred)
        return metrics
