from __future__ import annotations

import torch
import torch.nn.functional as F

from config import PAD_IDX, LABEL_SMOOTHING


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
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def token_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=PAD_IDX,
        label_smoothing=LABEL_SMOOTHING,
    )
