import copy
from collections import defaultdict

import torch
import torch.nn as nn
from lightning.fabric import Fabric


class GACFAS:
    def __init__(
        self,
        fabric: Fabric,
        model: nn.Module,
        rho: float = 0.5,
        eta: float = 0,
        alpha: float = 0.0005,
        n_domains: int = 3,
    ):
        self.fabric = fabric
        self.model = model
        self.rho = rho
        self.eta = eta
        self.alpha = alpha
        self.n_domains = n_domains
        self.proxy: dict[int, nn.Module] = {}
        self.state = defaultdict(dict)
        self.wgrad_norm = defaultdict(dict)

        for i in range(n_domains):
            self.proxy[i] = copy.deepcopy(model)
            self.state[i] = defaultdict(dict)
            self.wgrad_norm[i] = defaultdict(dict)

        self.accu_grad = None

    def reset_accum_grad(self):
        self.accu_grad = None
        for k in range(self.n_domains):
            self.proxy[k].zero_grad()

    @torch.no_grad()
    def get_perturb_norm(self, index: int):
        self.proxy[index].load_state_dict(self.model.state_dict())
        model_grad_dict = {
            name: param.grad.data.clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        if self.accu_grad == None:
            self.accu_grad = dict(
                zip([name for name in model_grad_dict], [0.0] * len(model_grad_dict))
            )

        wgrads = []
        for n, p in self.proxy[index].named_parameters():
            if n not in model_grad_dict:
                continue

            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(model_grad_dict[n] - self.accu_grad[n])

            t_w = self.state[index][n].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[index][n]["eps"] = t_w

            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))

        self.wgrad_norm[index] = torch.norm(torch.stack(wgrads), p=2) + 1.0e-16
        for k in self.accu_grad:
            self.accu_grad[k] += model_grad_dict[k]

    @torch.no_grad()
    def ascent_step(self, index: int):
        for n, p in self.proxy[index].named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[index][n].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / self.wgrad_norm[index])
            m_grad = self.accu_grad[n]
            eps.sub_(self.alpha * m_grad)
            p.add_(eps)

    def proxy_gradients(
        self,
        index: int,
        input: torch.Tensor,
        labels: torch.Tensor,
        loss_func: nn.Module,
        **kwargs
    ):
        self.proxy[index].train()
        logits = self.proxy[index](input, False)

        loss = loss_func(logits, labels, **kwargs)
        self.proxy[index].zero_grad()

        if isinstance(loss, list):
            loss = sum(loss)

        self.fabric.backward(loss)
        self.proxy[index].eval()

    @torch.no_grad()
    def descent_step(self):
        """
        Dont need to descent since we only need to obtain grad
        at ascending point of each domain, a.k.a., ascent_step
        """
        return None

    @torch.no_grad()
    def sync_grad_step(self, loss_list: list):
        proxies = [
            u
            for lo, u in zip(loss_list, [self.proxy[k] for k in range(self.n_domains)])
            if lo.item() != 0
        ]

        for params in zip(
            self.model.named_parameters(),
            *[proxy.named_parameters() for proxy in proxies]
        ):
            name_orig, param_orig = params[0]
            if param_orig.grad is None:
                continue

            avg_grad = sum(
                param.grad for name, param in params[1:] if param.grad is not None
            ) / len(proxies)

            if isinstance(avg_grad, torch.Tensor):
                param_orig.grad.data.add_(avg_grad.data.clone())
            else:
                param_orig.grad.add_(avg_grad)

        self.reset_accum_grad()
