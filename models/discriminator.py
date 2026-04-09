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
    FF_MULT,
    DISC_DROPOUT,
    PROJECTION_DIM,
)
from utils import build_valid_mask_from_lengths


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_proj = spectral_norm(nn.Linear(VOCAB_SIZE, DISC_EMBED_DIM))
        self.length_embedding = nn.Embedding(MAX_AA_LEN + 1, DISC_EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, DISC_EMBED_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DISC_EMBED_DIM,
            nhead=DISC_NUM_HEADS,
            dim_feedforward=DISC_EMBED_DIM * FF_MULT,
            dropout=DISC_DROPOUT,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=DISC_NUM_LAYERS)

        self.seq_head = spectral_norm(nn.Linear(DISC_EMBED_DIM, 1))
        self.seq_proj_head = spectral_norm(nn.Linear(DISC_EMBED_DIM, PROJECTION_DIM))
        self.cond_proj = spectral_norm(nn.Linear(ESM_DIM + DISC_EMBED_DIM, PROJECTION_DIM))

    def masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        maskf = mask.unsqueeze(-1).float()
        x = x * maskf
        summed = x.sum(dim=1)
        denom = maskf.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(self, toxin_emb: torch.Tensor, antidote_onehot: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        target_lengths = target_lengths.clamp(min=1, max=MAX_AA_LEN)
        valid_mask = build_valid_mask_from_lengths(target_lengths, antidote_onehot.size(1))
        x = self.seq_proj(antidote_onehot)
        x = x + self.pos_embedding[:, :antidote_onehot.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~valid_mask)
        seq_repr = self.masked_mean_pool(x, valid_mask)
        length_repr = self.length_embedding(target_lengths)
        cond = torch.cat([toxin_emb, length_repr], dim=-1)

        unconditional = self.seq_head(seq_repr)
        proj_seq = self.seq_proj_head(seq_repr)
        proj_cond = self.cond_proj(cond)
        projection = (proj_seq * proj_cond).sum(dim=-1, keepdim=True)
        return unconditional + projection
