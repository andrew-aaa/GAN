"""Метрики качества генерации: валидность, diversity, KL по аминокислотам, длина, EOS-структура."""

from __future__ import annotations

from typing import Iterable

import torch

from config import PAD_IDX, EOS_IDX, AA_START_IDX, VOCAB_SIZE
from utils import infer_lengths_from_tokens


def _valid_aa_tokens(token_ids: torch.Tensor):
    return token_ids[(token_ids >= AA_START_IDX) & (token_ids < VOCAB_SIZE)]


def amino_acid_frequency(token_ids: torch.Tensor) -> torch.Tensor:
    aa_tokens = _valid_aa_tokens(token_ids)
    counts = torch.ones(VOCAB_SIZE - AA_START_IDX, dtype=torch.float32, device=token_ids.device)
    if aa_tokens.numel() > 0:
        aa_tokens = aa_tokens - AA_START_IDX
        bincount = torch.bincount(aa_tokens, minlength=VOCAB_SIZE - AA_START_IDX).float()
        counts = counts + bincount
    return counts / counts.sum()


def aa_frequency_kl(fake_token_ids: torch.Tensor, real_token_ids: torch.Tensor) -> float:
    p = amino_acid_frequency(fake_token_ids)
    q = amino_acid_frequency(real_token_ids)
    return float((p * (p.log() - q.log())).sum().item())


def _sequence_without_special(token_row: Iterable[int]):
    seq = []
    for token in token_row:
        t = int(token)
        if t in (PAD_IDX, EOS_IDX):
            break
        if t >= AA_START_IDX:
            seq.append(t)
    return seq


def ngram_diversity(token_ids: torch.Tensor, n: int = 2) -> float:
    grams = []
    for row in token_ids.tolist():
        seq = _sequence_without_special(row)
        if len(seq) < n:
            continue
        grams.extend(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def eos_exact_rate(token_ids: torch.Tensor, target_lengths: torch.Tensor) -> float:
    ok = 0
    total = token_ids.size(0)
    for row, target_len in zip(token_ids, target_lengths):
        target_len = int(target_len)
        if target_len < row.numel() and int(row[target_len]) == EOS_IDX:
            ok += 1
    return ok / max(1, total)


def valid_eos_pad_rate(token_ids: torch.Tensor) -> float:
    ok = 0
    total = token_ids.size(0)
    for row in token_ids.tolist():
        eos_positions = [i for i, token in enumerate(row) if int(token) == EOS_IDX]
        if len(eos_positions) != 1:
            continue
        eos_pos = eos_positions[0]
        tail = row[eos_pos + 1:]
        if all(int(token) == PAD_IDX for token in tail):
            ok += 1
    return ok / max(1, total)


def length_mae(token_ids: torch.Tensor, target_lengths: torch.Tensor) -> float:
    pred_lengths = infer_lengths_from_tokens(token_ids)
    return float((pred_lengths.float() - target_lengths.float()).abs().mean().item())


def nonempty_ratio(token_ids: torch.Tensor) -> float:
    lengths = infer_lengths_from_tokens(token_ids)
    return float((lengths > 0).float().mean().item())


def repeat_ratio(token_ids: torch.Tensor) -> float:
    repeats = []
    for row in token_ids.tolist():
        seq = _sequence_without_special(row)
        if len(seq) < 2:
            repeats.append(0.0)
            continue
        rep = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1]) / (len(seq) - 1)
        repeats.append(rep)
    return float(sum(repeats) / max(1, len(repeats)))
