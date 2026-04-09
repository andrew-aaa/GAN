"""Функции потерь для гибридного обучения генератора и критика."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import PAD_IDX, PAD_WEIGHT


def gradient_penalty(discriminator, toxin_emb, real_samples, fake_samples, target_lengths, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolated = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(toxin_emb, interpolated, target_lengths)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def token_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = PAD_IDX, pad_weight: float = PAD_WEIGHT):
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    weights = torch.ones_like(targets, dtype=logits.dtype, device=logits.device)
    weights = torch.where(targets == pad_idx, torch.full_like(weights, pad_weight), weights)

    return (loss * weights).sum() / weights.sum().clamp(min=1.0)
