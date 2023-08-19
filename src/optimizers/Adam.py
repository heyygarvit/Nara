import sys
sys.path.append('src')
from core.GPU import cp

class Adam:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, amsgrad=False, clip_value=None, warmup_steps=1000, decay_factor=0.1, decay_steps=10000, logging=True, max_logs=1000):
        self.parameters = parameters
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.logging = logging
        self.max_logs = max_logs
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.m = [cp.zeros_like(param.data) for param in self.parameters]
        self.v = [cp.zeros_like(param.data) for param in self.parameters]
        self.v_max = [cp.zeros_like(param.data) for param in self.parameters] if amsgrad else None
        self.t = 0
        self.log_data = []

    


    def clip_gradients(self, gradient):
        if self.clip_value:
            return cp.clip(gradient, -self.clip_value, self.clip_value)
        return gradient

    def warmup_lr(self):
        if self.t < self.warmup_steps:
            self.learning_rate = self.init_learning_rate * (self.t / self.warmup_steps)

    def step_decay(self):
        if self.t % self.decay_steps == 0:
            self.learning_rate *= self.decay_factor

    def log(self, gradient_norm, param_norm):
        if not self.logging:
            return
        self.log_data.append({"step": self.t, "gradient_norm": gradient_norm, "param_norm": param_norm, "learning_rate": self.learning_rate})
        # Limit the number of logs stored
        if len(self.log_data) > self.max_logs:
            self.log_data.pop(0)

    def step(self):
        self.t += 1
        self.warmup_lr()
        self.step_decay()

        for idx, param in enumerate(self.parameters):
            g = self.clip_gradients(param.grad.data)
            g = cp.array(g)

            # Update biased first moment estimate
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g**2

            if self.v_max:
                self.v_max[idx] = cp.maximum(self.v_max[idx], self.v[idx])

            # Compute bias-corrected first moment estimate
            m_corrected = self.m[idx] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v_max[idx] if self.v_max else self.v[idx] / (1 - self.beta2**self.t)

            update = self.learning_rate * m_corrected / (cp.sqrt(v_corrected) + self.epsilon)
            param.data -= update + self.weight_decay * param.data


            # Logging
            gradient_norm = cp.linalg.norm(g)
            param_norm = cp.linalg.norm(param.data)
            self.log(gradient_norm, param_norm)


    def zero_grad(self):
        for param in self.parameters:
            param.grad.data = cp.zeros_like(param.data)