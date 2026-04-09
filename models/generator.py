from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    LATENT_DIM,
    MAX_AA_LEN,
    MAX_LEN,
    VOCAB_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    ESM_DIM,
    FF_MULT,
    DROPOUT,
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
)
from utils import gumbel_softmax


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.toxin_proj = nn.Linear(ESM_DIM, EMBED_DIM)
        self.noise_proj = nn.Linear(LATENT_DIM, EMBED_DIM)
        self.length_embedding = nn.Embedding(MAX_AA_LEN + 1, EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, EMBED_DIM) * 0.02)

        self.length_predictor = nn.Sequential(
            nn.Linear(ESM_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, MAX_AA_LEN + 1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * FF_MULT,
            dropout=DROPOUT,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def get_length_logits(self, toxin_emb: torch.Tensor) -> torch.Tensor:
        return self.length_predictor(toxin_emb)

    @torch.no_grad()
    def predict_lengths(self, toxin_emb: torch.Tensor) -> torch.Tensor:
        return self.get_length_logits(toxin_emb).argmax(dim=-1).clamp(min=1, max=MAX_AA_LEN)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        m = torch.full((seq_len, seq_len), float('-inf'), device=device)
        return torch.triu(m, diagonal=1)

    def _build_condition(self, toxin_emb: torch.Tensor, z: Optional[torch.Tensor], target_lengths: Optional[torch.Tensor]):
        if z is None:
            z = torch.randn(toxin_emb.size(0), LATENT_DIM, device=toxin_emb.device)
        if target_lengths is None:
            target_lengths = self.predict_lengths(toxin_emb)
        target_lengths = target_lengths.clamp(min=1, max=MAX_AA_LEN)
        cond = self.toxin_proj(toxin_emb) + self.noise_proj(z) + self.length_embedding(target_lengths)
        return cond, target_lengths

    def forward_teacher(
        self,
        decoder_input: torch.Tensor,
        toxin_emb: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ):
        cond, target_lengths = self._build_condition(toxin_emb, z, target_lengths)
        tok = self.token_embedding(decoder_input)
        x = tok + cond.unsqueeze(1) + self.pos_embedding[:, :decoder_input.size(1), :]
        x = self.dropout(x)
        hidden = self.transformer(x, mask=self._causal_mask(decoder_input.size(1), decoder_input.device))
        logits = self.fc_out(hidden)
        return logits, target_lengths

    def sample(
        self,
        toxin_emb: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = True,
    ):
        cond, target_lengths = self._build_condition(toxin_emb, z, target_lengths)
        bsz = toxin_emb.size(0)
        device = toxin_emb.device
        generated = torch.full((bsz, MAX_LEN), PAD_IDX, dtype=torch.long, device=device)
        generated[:, 0] = BOS_IDX
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        probs_steps = []
        for step in range(MAX_LEN):
            logits, _ = self.forward_teacher(generated, toxin_emb, z=z, target_lengths=target_lengths)
            step_logits = logits[:, step, :] / max(temperature, 1e-4)

            # EOS/PAD enforcement by target length
            before = step < target_lengths
            at = step == target_lengths
            after = step > target_lengths
            if before.any():
                step_logits[before, EOS_IDX] = -1e9
                step_logits[before, PAD_IDX] = -1e9
            if at.any():
                step_logits[at] = -1e9
                step_logits[at, EOS_IDX] = 0.0
            if after.any():
                step_logits[after] = -1e9
                step_logits[after, PAD_IDX] = 0.0

            step_probs = gumbel_softmax(step_logits, temperature=1.0, hard=hard)
            probs_steps.append(step_probs.unsqueeze(1))
            step_ids = step_probs.argmax(dim=-1)

            if step + 1 < MAX_LEN:
                step_ids = torch.where(finished, torch.full_like(step_ids, PAD_IDX), step_ids)
                generated[:, step + 1] = step_ids
                finished = finished | (step_ids == EOS_IDX)

        probs = torch.cat(probs_steps, dim=1)
        return probs, target_lengths
