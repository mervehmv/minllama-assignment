from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                weight_decay = group["weight_decay"]
                # Update first and second moments of the gradients
                if "step" not in state:
                    state["step"] = 0  # Initialize if first update
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)  # First moment (m)
                    state["exp_avg_sq"] = torch.zeros_like(grad)  # Second moment (v)

                
                state["step"] += 1  # Increment step count
                t = state["step"]  # Get current timestep
                m, v = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"][0], group["betas"][1]

                m.mul_(beta1)
                m.add_(grad, alpha=1 - beta1)

                v.mul_(beta2)
                v.addcmul_(grad, grad, value=1 - beta2)

                state["exp_avg"], state["exp_avg_sq"] = m, v


                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                #m_head = m/(1-beta1**t)
                #v_head = v/(1-beta2**t)
                alpha_t = alpha * math.sqrt(1-beta2**t) / (1-beta1**t)

                # Update parameters
                
                #self.params = self.params - m_head.mul_(alpha)/torch.sqrt(v_head)+self.eps

                p.data.mul_(1 - alpha_t * weight_decay)  # Apply weight decay
                p.data.addcdiv_(m, (v.sqrt() + group["eps"]), value=-alpha_t)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss