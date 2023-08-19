
import sys
sys.path.append('src')
from core.GPU import cp


class GradientDescent:
    def __init__(self, learning_rate=0.01, momentum=0.9, decay_rate=0.95, max_gradient_norm=None, use_nesterov=False, l2_reg_strength=0.0, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.global_step = 0
        self.velocity = {}
        self.max_gradient_norm = max_gradient_norm
        self.use_nesterov = use_nesterov
        self.l2_reg_strength = l2_reg_strength
        self.rmsprop_cache = {} 
        self.beta = beta
        self.epsilon = epsilon

    def xavier_initialization(self, shape):
        return cp.random.randn(*shape) * cp.sqrt(2. / sum(shape))

    def he_initialization(self, shape):
        return cp.random.randn(*shape) * cp.sqrt(2. / shape[0])

    def adaptive_learning_rate(self, gradient, param_name):
        if param_name not in self.rmsprop_cache:
            self.rmsprop_cache[param_name] = cp.zeros_like(gradient)
        self.rmsprop_cache[param_name] = self.beta * self.rmsprop_cache[param_name] + (1 - self.beta) * gradient**2
        adjusted_gradient = gradient / (cp.sqrt(self.rmsprop_cache[param_name]) + self.epsilon)
        return adjusted_gradient

    def clip_gradients(self, gradients):
        if self.max_gradient_norm:
            total_norm = 0
            for gradient in gradients.values():
                total_norm += (gradient**2).sum()
            total_norm = total_norm**0.5
            clip_coef = self.max_gradient_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for param_name in gradients:
                    gradients[param_name] = gradients[param_name] * clip_coef
        return gradients

    def step(self, gradients, params):
        gradients = self.clip_gradients(gradients)
        lr_t = self.learning_rate * (self.decay_rate ** (self.global_step / 1000))
        self.global_step += 1

        updated_params = {}
        for param_name, gradient in gradients.items():
            # Apply L2 regularization
            gradient += self.l2_reg_strength * params[param_name]
            gradient = self.adaptive_learning_rate(gradient, param_name)

            if param_name not in self.velocity:
                self.velocity[param_name] = 0
            self.velocity[param_name] = self.momentum * self.velocity[param_name] + (1 - self.momentum) * gradient

            if self.use_nesterov:
                nesterov_gradient = gradient + self.momentum * self.velocity[param_name]
                updated_params[param_name] = params[param_name] - lr_t * nesterov_gradient
            else:
                updated_params[param_name] = params[param_name] - lr_t * self.velocity[param_name]
        return updated_params

    def zero_grad(self):
        self.velocity = {}
        self.rmsprop_cache = {}
