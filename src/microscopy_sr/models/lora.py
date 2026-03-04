from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import SelfAttention


class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear with a low-rank additive update: W = W0 + scale * B @ A.

    The original weights are frozen. Only lora_A and lora_B are trainable.
    B is zero-initialized so the module is identical to the base linear at the
    start of adaptation.
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, device=device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device))
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scale * lora


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
) -> nn.Module:
    """
    Injects LoRA into every SelfAttention module in *model* (in-place).

    Replaces the .qkv and .out_proj nn.Linear layers with LoRALinear.
    Call this AFTER loading pretrained weights and BEFORE saving/loading
    a LoRA checkpoint.

    Returns the modified model.
    """
    for module in model.modules():
        if isinstance(module, SelfAttention):
            module.qkv = LoRALinear(module.qkv, rank=rank, alpha=alpha)
            module.out_proj = LoRALinear(module.out_proj, rank=rank, alpha=alpha)
    return model


def freeze_non_lora(model: nn.Module) -> None:
    """
    Freezes all parameters that are not LoRA matrices.

    After calling this, only lora_A and lora_B parameters have requires_grad=True,
    making them the sole targets for the optimizer during adaptation.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name
