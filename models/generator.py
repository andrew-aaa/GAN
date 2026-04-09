"""Length-aware генератор с teacher forcing, autoregressive sampling и строгим EOS/PAD-контролем."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import (
    LATENT_DIM,
    MAX_AA_LEN,
    MAX_LEN,
    VOCAB_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    ESM_DIM,
    DROPOUT,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    AA_START_IDX,
)
from utils import gumbel_softmax


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.toxin_proj = nn.Linear(ESM_DIM, EMBED_DIM)
        self.noise_proj = nn.Linear(LATENT_DIM, EMBED_DIM)
        self.length_embedding = nn.Embedding(MAX_AA_LEN + 1, EMBED_DIM)

        self.length_predictor = nn.Sequential(
            nn.Linear(ESM_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, MAX_AA_LEN + 1),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, EMBED_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.dropout = nn.Dropout(DROPOUT)
        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def get_length_logits(self, toxin_emb: torch.Tensor) -> torch.Tensor:
        return self.length_predictor(toxin_emb)

    @torch.no_grad()
    def predict_lengths(self, toxin_emb: torch.Tensor) -> torch.Tensor:
        logits = self.get_length_logits(toxin_emb)
        lengths = torch.argmax(logits, dim=-1)
        return lengths.clamp(min=1, max=MAX_AA_LEN)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _build_condition(
        self,
        toxin_emb: torch.Tensor,
        z: Optional[torch.Tensor],
        target_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = toxin_emb.size(0)
        device = toxin_emb.device

        if z is None:
            z = torch.zeros(batch_size, LATENT_DIM, device=device)
        if target_lengths is None:
            target_lengths = self.predict_lengths(toxin_emb)
        else:
            target_lengths = target_lengths.clamp(min=1, max=MAX_AA_LEN)

        cond = self.toxin_proj(toxin_emb) + self.noise_proj(z) + self.length_embedding(target_lengths)
        return cond, target_lengths

    def forward_teacher(
        self,
        decoder_input_ids: torch.Tensor,
        toxin_emb: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing: decoder_input_ids уже содержит [BOS, aa1, aa2, ...]."""
        cond, target_lengths = self._build_condition(toxin_emb, z, target_lengths)
        seq_len = decoder_input_ids.size(1)
        device = decoder_input_ids.device

        x = self.token_embedding(decoder_input_ids)
        x = x + cond.unsqueeze(1) + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        x = self.transformer(x, mask=self._causal_mask(seq_len, device))
        logits = self.fc_out(x)
        return logits, target_lengths

    def _mask_step_logits(self, step_logits: torch.Tensor, step: int, target_lengths: torch.Tensor) -> torch.Tensor:
        """Строгий контроль структуры: до длины — только AA, затем EOS, затем PAD."""
        masked = step_logits.clone()

        before = step < target_lengths
        at_eos = step == target_lengths
        after = step > target_lengths

        if before.any():
            idx = before.nonzero(as_tuple=False).squeeze(1)
            masked[idx, :AA_START_IDX] = float("-inf")

        if at_eos.any():
            idx = at_eos.nonzero(as_tuple=False).squeeze(1)
            forced = torch.full_like(masked[idx], float("-inf"))
            forced[:, EOS_IDX] = 0.0
            masked[idx] = forced

        if after.any():
            idx = after.nonzero(as_tuple=False).squeeze(1)
            forced = torch.full_like(masked[idx], float("-inf"))
            forced[:, PAD_IDX] = 0.0
            masked[idx] = forced

        return masked

    def sample(
        self,
        toxin_emb: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        sampling_temperature: float = 1.0,
        gumbel_tau: float = 1.0,
        hard: bool = True,
        return_logits: bool = False,
    ):
        """
        Autoregressive генерация.

        sampling_temperature масштабирует логиты и реально влияет на argmax-путь даже при hard=True.
        gumbel_tau управляет "мягкостью" gumbel-softmax релаксации.
        """
        cond, target_lengths = self._build_condition(toxin_emb, z, target_lengths)
        batch_size = toxin_emb.size(0)
        device = toxin_emb.device

        input_embeds = torch.zeros(batch_size, MAX_LEN, EMBED_DIM, device=device)
        bos_embed = self.token_embedding.weight[BOS_IDX].unsqueeze(0).expand(batch_size, -1)
        input_embeds[:, 0, :] = bos_embed

        output_probs = torch.zeros(batch_size, MAX_LEN, VOCAB_SIZE, device=device)
        collected_logits = []
        causal_mask = self._causal_mask(MAX_LEN, device)

        for step in range(MAX_LEN):
            x = input_embeds + cond.unsqueeze(1) + self.pos_embedding[:, :MAX_LEN, :]
            x = self.dropout(x)
            hidden = self.transformer(x, mask=causal_mask)
            logits = self.fc_out(hidden)
            step_logits = logits[:, step, :]
            step_logits = self._mask_step_logits(step_logits, step, target_lengths)
            scaled_logits = step_logits / max(float(sampling_temperature), 1e-6)

            step_probs = gumbel_softmax(scaled_logits, temperature=gumbel_tau, hard=hard)
            output_probs[:, step, :] = step_probs
            collected_logits.append(step_logits.unsqueeze(1))

            if step + 1 < MAX_LEN:
                next_embed = step_probs @ self.token_embedding.weight
                input_embeds[:, step + 1, :] = next_embed

        if return_logits:
            return output_probs, target_lengths, torch.cat(collected_logits, dim=1)
        return output_probs, target_lengths
