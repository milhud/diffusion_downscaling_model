"""Exponential Moving Average for model parameters."""

import copy
import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)
        # After each optimizer step:
        ema.update()
        # For evaluation:
        with ema.apply():
            model(x)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self):
        return _EMAContext(self.model, self.shadow)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class _EMAContext:
    """Context manager to temporarily swap model params with EMA params."""

    def __init__(self, model, shadow):
        self.model = model
        self.shadow = shadow
        self.backup = {}

    def __enter__(self):
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return self.model

    def __exit__(self, *args):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
