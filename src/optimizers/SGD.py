import sys
sys.path.append('src')
from core.GPU import cp
from core.Tensor import Tensor

class SGD:
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9, weight_decay=0, nesterov=False, lr_decay_factor=0.1, lr_decay_steps=10000, grad_clip=None, noise_stddev=0.0, custom_lr_decay=None, logging=True, max_logs=1000):
        """
        Stochastic Gradient Descent (SGD) optimizer with various enhancements.

        :param parameters: Model parameters to optimize
        :param learning_rate: Initial learning rate
        :param momentum: Momentum factor
        :param weight_decay: Weight decay (L2 regularization) factor
        :param nesterov: Use Nesterov momentum if True
        :param lr_decay_factor: Learning rate decay factor
        :param lr_decay_steps: Number of steps before applying learning rate decay
        :param grad_clip: Gradient clipping threshold
        :param noise_stddev: Standard deviation of Gaussian noise to inject into gradients
        :param custom_lr_decay: A custom function for learning rate decay (should accept the current step and initial learning rate as arguments)
        """
        
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_steps = lr_decay_steps
        self.grad_clip = grad_clip
        self.noise_stddev = noise_stddev
        self.custom_lr_decay = custom_lr_decay
        self.logging = logging
        self.max_logs = max_logs
        self.t = 0
        self.velocity = [cp.zeros_like(param.data) for param in self.parameters]
        self.log_data = []

    # def step(self):
    #     """
    #     Performs a single optimization step.
    #     """
    #     self.t += 1
        
    #     # Custom LR decay if provided, else use the default step decay
    #     if self.custom_lr_decay:
    #         lr_t = self.custom_lr_decay(self.t, self.learning_rate)
    #     else:
    #         lr_t = self.learning_rate * (self.lr_decay_factor ** (self.t // self.lr_decay_steps))

    def step(self):
        for idx, param in enumerate(self.parameters):
            if param.requires_grad:
                # Ensure param.grad is a Tensor
                if isinstance(param.grad, Tensor):
                    grad = param.grad.data + self.weight_decay * param.data

                    # Gradient Clipping
                    if self.grad_clip:
                        grad = cp.clip(grad, -self.grad_clip, self.grad_clip)

                    # Noise Injection
                    if self.noise_stddev:
                        grad += cp.random.normal(0, self.noise_stddev, size=grad.shape)

                    # Momentum
                    self.velocity[idx] = self.momentum * self.velocity[idx] - self.learning_rate * grad
                    # print("Shape of param.data:", param.data.shape)
                    # print("Shape of self.velocity[idx]:", self.velocity[idx].shape)
                    # print("Shape of grad:", grad.shape)
                    if param.data.shape != self.velocity[idx].shape:
                        self.velocity[idx] = self.velocity[idx].squeeze()

                    # Nesterov Momentum
                    if self.nesterov:
                        param.data += self.momentum * self.velocity[idx] - self.learning_rate * grad
                    else:
                        param.data += self.velocity[idx]

                    # Logging
                    if self.logging:
                        gradient_norm = cp.linalg.norm(grad)
                        self.log_data.append({"step": self.t, "gradient_norm": gradient_norm, "learning_rate": self.learning_rate})
                        # Limit the number of logs stored
                        if len(self.log_data) > self.max_logs:
                            self.log_data.pop(0)
                else:
                    print(f"Warning: Gradient for parameter {param} is not a Tensor. Skipping update.")


    def zero_grad(self):
        for param in self.parameters:
            param.grad.data = cp.zeros_like(param.data)
           


