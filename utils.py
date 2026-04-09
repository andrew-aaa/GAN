"""Вспомогательные функции для кодирования и декодирования белковых последовательностей."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from config import (
    VOCAB,
    MAX_AA_LEN,
    MAX_LEN,
    PAD_IDX,
    PAD_TOKEN,
    BOS_IDX,
    BOS_TOKEN,
    EOS_IDX,
    EOS_TOKEN,
)


aa_to_idx = {aa: i for i, aa in enumerate(VOCAB)}
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}


def clean_sequence(seq: str) -> str:
    """Оставляет только допустимые аминокислоты и удаляет служебные токены."""
    seq = seq.strip().upper()
    return "".join(
        aa for aa in seq
        if aa in aa_to_idx and aa not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)
    )


def encode_sequence(seq: str, max_aa_len: int = MAX_AA_LEN, max_len: int = MAX_LEN):
    """
    Преобразует строку в пару для teacher forcing.

    Возвращает:
      decoder_input: [BOS, aa1, aa2, ..., aaN, PAD, ...]
      target:        [aa1, aa2, ..., aaN, EOS, PAD, ...]
      aa_length:     число аминокислот без EOS/PAD
    """
    seq = clean_sequence(seq)[:max_aa_len]
    aa_length = len(seq)

    aa_ids = [aa_to_idx[aa] for aa in seq]
    decoder_input = [BOS_IDX] + aa_ids
    target = aa_ids + [EOS_IDX]

    if len(decoder_input) < max_len:
        decoder_input += [PAD_IDX] * (max_len - len(decoder_input))
    else:
        decoder_input = decoder_input[:max_len]

    if len(target) < max_len:
        target += [PAD_IDX] * (max_len - len(target))
    else:
        target = target[:max_len]
        target[-1] = EOS_IDX
        aa_length = min(aa_length, max_len - 1)

    return decoder_input, target, aa_length


def decode_sequence(indices: Iterable[int]) -> str:
    """Собирает аминокислотную строку до EOS/PAD."""
    chars = []
    for i in indices:
        i = int(i)
        if i in (PAD_IDX, BOS_IDX, EOS_IDX):
            if i == EOS_IDX or i == PAD_IDX:
                break
            continue
        aa = idx_to_aa.get(i)
        if aa is not None:
            chars.append(aa)
    return "".join(chars)


def to_one_hot(seq: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(seq, num_classes=vocab_size).float()


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


def build_length_masks(target_lengths: torch.Tensor, seq_len: int):
    """
    target_lengths: [B] — число аминокислот до EOS.
    EOS должен стоять в позиции == target_length.
    """
    device = target_lengths.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    target_lengths = target_lengths.unsqueeze(1)

    before_eos = positions < target_lengths
    eos_pos = positions == target_lengths
    after_eos = positions > target_lengths
    return before_eos, eos_pos, after_eos


def infer_lengths_from_tokens(token_ids: torch.Tensor, eos_idx: int = EOS_IDX, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """Оценивает длину как число токенов до первого EOS/PAD."""
    lengths = []
    for row in token_ids:
        pos = 0
        while pos < row.numel() and int(row[pos]) not in (eos_idx, pad_idx):
            pos += 1
        lengths.append(pos)
    return torch.tensor(lengths, device=token_ids.device, dtype=torch.long)
