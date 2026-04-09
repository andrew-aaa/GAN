"""EMA-обёртка для генератора."""

from __future__ import annotations

import copy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for key, value in ssd.items():
            if key not in msd:
                continue
            src = msd[key].detach()
            if not torch.is_floating_point(src):
                value.copy_(src)
            else:
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()
