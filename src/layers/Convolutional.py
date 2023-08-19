import numpy as np
import sys
sys.path.append('src')
from core.Tensor import Tensor
from core.GPU import cp



class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding='SAME', activation='relu', init_method='he', reg_strength=0.01):
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.reg = reg_strength

        if init_method == 'he':
            init_val = np.sqrt(2. / (input_channels * kernel_size * kernel_size))
        elif init_method == 'xavier':
            init_val = np.sqrt(1. / (input_channels * kernel_size * kernel_size))
        else:
            raise ValueError("Initialization type not recognized")
        
        self.filters = cp.random.randn(output_channels, input_channels, kernel_size, kernel_size) * init_val
        self.biases = cp.zeros((output_channels, 1, 1, 1))
        self.gamma = cp.ones((output_channels, 1, 1, 1))
        self.beta = cp.zeros((output_channels, 1, 1, 1))
      
        self.moving_mean = cp.zeros((1, 16, 1, 1))
        self.moving_variance = cp.ones((1, 16, 1, 1))

        self.training = True

    def batchnorm_forward(self, x):
        epsilon = 1e-5
        if self.training:
           
            mean = cp.squeeze(cp.mean(x, axis=(0, 2, 3), keepdims=True), axis=0)
            variance = cp.squeeze(cp.var(x, axis=(0, 2, 3), keepdims=True), axis=0)
               # Add the missing dimension
            mean = mean[np.newaxis, :, :, :]
            variance = variance[np.newaxis, :, :, :]
            x_normalized = (x - mean) / cp.sqrt(variance + epsilon)
            out = self.gamma.reshape(1, -1, 1, 1) * x_normalized + self.beta.reshape(1, -1, 1, 1)
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
            self.moving_variance = 0.9 * self.moving_variance + 0.1 * variance
        else:
            x_normalized = (x - self.moving_mean) / cp.sqrt(self.moving_variance + epsilon)
            out = self.gamma.reshape(1, -1, 1, 1) * x_normalized + self.beta.reshape(1, -1, 1, 1)
        return out
    
    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h) // stride + 1
        out_w = (W + 2*pad - filter_w) // stride + 1

        img_padded = cp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img_padded[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h) // stride + 1
        out_w = (W + 2*pad - filter_w) // stride + 1

        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        img = cp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]
    
    def forward(self, input_data):
        self.input = input_data
        N, C, H, W = input_data.shape
        FN, C, FH, FW = self.filters.shape

        if self.padding == 'SAME':
            pad = (FH - 1) // 2
        elif self.padding == 'VALID':
            pad = 0
        else:
            raise ValueError("Padding should be 'SAME' or 'VALID'")

        out_h = (H + 2*pad - FH) // self.stride + 1
        out_w = (W + 2*pad - FW) // self.stride + 1

        col = self.im2col(input_data, FH, FW, self.stride, pad)
        col_filters = self.filters.reshape(FN, -1).T
        out = cp.dot(col, col_filters) + self.biases.reshape(1, -1)  # Change here

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        out = self.batchnorm_forward(out)

        if self.activation == 'relu':
            out = self.relu(out)
        elif self.activation == 'leaky_relu':
            out = self.leaky_relu(out)
        elif self.activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.activation == 'tanh':
            out = self.tanh(out)

        self.output = out 
        return out
    
    def relu(self, x):
        return cp.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        return cp.maximum(alpha * x, x)

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))

    def tanh(self, x):
        return (2 / (1 + cp.exp(-2*x))) - 1

    def backward(self, dout):
        FN, C, FH, FW = self.filters.shape

        if self.activation == 'relu':
            dout *= (self.output > 0)
        elif self.activation == 'leaky_relu':
            dout *= cp.where(self.output > 0, 1, 0.01)

        # Gradient with respect to batch normalization
        dgamma, dbeta, dx_normalized = self.batchnorm_backward(dout)
        self.dgamma = dgamma
        self.dbeta = dbeta

        # Gradient with respect to convolution
        if self.padding == 'SAME':
            pad = (FH - 1) // 2
        elif self.padding == 'VALID':
            pad = 0
        else:
            raise ValueError("Padding should be 'SAME' or 'VALID'")

        col = self.im2col(self.input, FH, FW, self.stride, pad)
        col_filters = self.filters.reshape(FN, -1).T

       
        dcol = cp.dot(dx_normalized.reshape(-1, FN), col_filters.T)


        self.dfilters = cp.dot(col.T, dx_normalized.reshape(-1, FN)).transpose(1, 0).reshape(FN, C, FH, FW)
        self.dbiases = cp.sum(dx_normalized, axis=(0, 2, 3)).reshape(FN, 1, 1, 1)

        dinput = self.col2im(dcol, self.input.shape, FH, FW, self.stride, pad)

        # L2 regularization for filters
        self.dfilters += self.reg * self.filters

        return dinput

    def batchnorm_backward(self, dout):
        N, C, H, W = dout.shape
       
        x_normalized = (self.output - self.moving_mean.reshape(1, 16, 1, 1)) / cp.sqrt(self.moving_variance.reshape(1, 16, 1, 1) + 1e-5)


        dbeta = cp.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
        dgamma = cp.sum(dout * x_normalized, axis=(0, 2, 3)).reshape(1, C, 1, 1)
        dx_normalized = dout * self.gamma.reshape(1, C, 1, 1)

        dvariance = cp.sum(dx_normalized * (self.output - self.moving_mean) * -0.5 * cp.power(self.moving_variance + 1e-5, -1.5), axis=(0, 2, 3)).reshape(1, C, 1, 1)
        dmean = cp.sum(dx_normalized * -1 / cp.sqrt(self.moving_variance + 1e-5), axis=(0, 2, 3)).reshape(1, C, 1, 1) + dvariance * cp.sum(-2 * (self.output - self.moving_mean), axis=(0, 2, 3)).reshape(1, C, 1, 1) / N

        dx = (dx_normalized / cp.sqrt(self.moving_variance + 1e-5)) + (dvariance * 2 * (self.output - self.moving_mean) / N) + (dmean / N)
        return dgamma, dbeta, dx