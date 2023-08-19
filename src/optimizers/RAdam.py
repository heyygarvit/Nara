import sys
sys.path.append('src')
from core.GPU import cp

class RAdam:
    def __init__(self, parameters, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, warmup_steps=1000, decay_factor=0.1, decay_steps=10000, logging=True, max_logs=1000):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.logging = logging
        self.max_logs = max_logs
        self.no_improve_count = 0
        self.best_loss = float('inf')
        self.m = [cp.zeros_like(p.data) for p in self.parameters]
        self.v = [cp.zeros_like(p.data) for p in self.parameters]
        self.v_max = [cp.zeros_like(p.data) for p in self.parameters] if amsgrad else None
        self.t = 0
        self.log_data = []

    def warmup_lr(self):
        if self.t < self.warmup_steps:
            return self.learning_rate * (self.t / self.warmup_steps)
        return self.learning_rate

    def step_decay(self):
        if self.t % self.decay_steps == 0:
            self.learning_rate *= self.decay_factor

    def log(self, gradient_norm, param_norm, loss=None):
        if not self.logging:
            return
        log_entry = {"step": self.t, "gradient_norm": gradient_norm, "param_norm": param_norm, "learning_rate": self.learning_rate}
        if loss:
            log_entry["loss"] = loss
        self.log_data.append(log_entry)
        # Limit the number of logs stored
        if len(self.log_data) > self.max_logs:
            self.log_data.pop(0)


    def step(self, gradient, loss=None):
        self.t += 1
        lr_t = self.warmup_lr()
        self.step_decay()

        beta1, beta2 = self.betas

        rho_inf = 2.0 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * self.t * beta2 ** self.t / (1 - beta2 ** self.t)

        for idx, (p, g) in enumerate(zip(self.parameters, gradient)):
            if self.weight_decay != 0:
                g += self.weight_decay * p.data

            self.m[idx] = beta1 * self.m[idx] + (1 - beta1) * g
            self.v[idx] = beta2 * self.v[idx] + (1 - beta2) * g * g

            if rho_t > 4:  # This threshold can vary but is commonly set to 4.
                rt = (rho_t - 4) * (rho_t - 2) * rho_inf / (rho_inf - 4) / (rho_inf - 2) / rho_t
                denom = cp.sqrt(self.v[idx] / (1 - beta2 ** self.t) + self.eps)
                update = lr_t * rt * self.m[idx] / (1 - beta1 ** self.t) / denom
            else:
                update = lr_t * self.m[idx] / (1 - beta1 ** self.t)

            p.data -= update

            # Logging
            gradient_norm = cp.linalg.norm(g)
            param_norm = cp.linalg.norm(p.data)
            self.log(gradient_norm, param_norm, loss)

 
    def zero_grad(self):
        for param in self.parameters:
            param.grad = cp.zeros_like(param.data)