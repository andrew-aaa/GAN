"""Projection-discriminator для пары (токсин, длина, антидот) с mask по длине."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from config import (
    MAX_AA_LEN,
    MAX_LEN,
    VOCAB_SIZE,
    ESM_DIM,
    DISC_EMBED_DIM,
    DISC_NUM_HEADS,
    DISC_NUM_LAYERS,
    DISC_DROPOUT,
    PROJECTION_DIM,
)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq_proj = spectral_norm(nn.Linear(VOCAB_SIZE, DISC_EMBED_DIM))
        self.length_embedding = nn.Embedding(MAX_AA_LEN + 1, DISC_EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, DISC_EMBED_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DISC_EMBED_DIM,
            nhead=DISC_NUM_HEADS,
            dim_feedforward=DISC_EMBED_DIM * 4,
            dropout=DISC_DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=DISC_NUM_LAYERS)

        # Признаки последовательности для projection-критика
        self.seq_feature = nn.Sequential(
            spectral_norm(nn.Linear(DISC_EMBED_DIM + DISC_EMBED_DIM, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, PROJECTION_DIM)),
            nn.LeakyReLU(0.2),
        )

        # Безусловный скаляр f(x)
        self.unconditional_head = spectral_norm(nn.Linear(PROJECTION_DIM, 1))

        # Условный вектор h(c)
        self.cond_proj = spectral_norm(nn.Linear(ESM_DIM, PROJECTION_DIM))

    def build_valid_mask(self, target_lengths: torch.Tensor) -> torch.Tensor:
        """True для валидных позиций [AA ... EOS], False для PAD после EOS."""
        positions = torch.arange(MAX_LEN, device=target_lengths.device).unsqueeze(0)
        return positions <= target_lengths.unsqueeze(1)

    def masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        x = x * mask
        summed = x.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(self, toxin_emb: torch.Tensor, antidote_onehot: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        target_lengths = target_lengths.clamp(min=1, max=MAX_AA_LEN)
        valid_mask = self.build_valid_mask(target_lengths)

        x = self.seq_proj(antidote_onehot)
        x = x + self.pos_embedding
        x = self.transformer(x, src_key_padding_mask=~valid_mask)

        seq_repr = self.masked_mean_pool(x, valid_mask)
        length_repr = self.length_embedding(target_lengths)
        seq_joint = torch.cat([seq_repr, length_repr], dim=1)

        seq_features = self.seq_feature(seq_joint)
        unconditional = self.unconditional_head(seq_features)
        cond_features = self.cond_proj(toxin_emb)
        projection = (seq_features * cond_features).sum(dim=1, keepdim=True)

        return unconditional + projection
