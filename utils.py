from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    return ''.join(aa for aa in seq if aa in aa_to_idx and aa not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN))


def encode_sequence(seq: str, max_aa_len: int = MAX_AA_LEN):
    seq = clean_sequence(seq)[:max_aa_len]
    aa_length = len(seq)
    aa_ids = [aa_to_idx[aa] for aa in seq]

    decoder_input = [BOS_IDX] + aa_ids
    target = aa_ids + [EOS_IDX]

    decoder_input = decoder_input[:MAX_LEN]
    target = target[:MAX_LEN]

    if len(decoder_input) < MAX_LEN:
        decoder_input += [PAD_IDX] * (MAX_LEN - len(decoder_input))
    if len(target) < MAX_LEN:
        target += [PAD_IDX] * (MAX_LEN - len(target))

    aa_length = min(aa_length, MAX_AA_LEN)
    return decoder_input, target, aa_length


def decode_sequence(indices: Iterable[int]) -> str:
    out = []
    for idx in indices:
        idx = int(idx)
        if idx in (PAD_IDX, BOS_IDX, EOS_IDX):
            if idx == EOS_IDX:
                break
            continue
        aa = idx_to_aa.get(idx)
        if aa is not None and aa not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            out.append(aa)
    return ''.join(out)


def to_one_hot(seq: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(seq, num_classes=vocab_size).float()


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


def build_valid_mask_from_lengths(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
    # valid target positions are amino acids [0..len-1] plus EOS at position len
    return positions <= lengths.unsqueeze(1)


def write_metrics_row(csv_path: str, fieldnames: list[str], row: dict) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
