from __future__ import annotations

from collections import Counter
import math

import torch

from config import PAD_IDX, EOS_IDX, AA_START_IDX
from utils import decode_sequence


def _trim_ids(row: torch.Tensor):
    out = []
    for x in row.tolist():
        if x in (PAD_IDX,):
            continue
        if x == EOS_IDX:
            break
        if x >= AA_START_IDX:
            out.append(int(x))
    return out


def nonempty_ratio(fake_ids: torch.Tensor) -> float:
    vals = [1.0 if len(_trim_ids(row)) > 0 else 0.0 for row in fake_ids]
    return float(sum(vals) / max(1, len(vals)))


def repeat_ratio(fake_ids: torch.Tensor) -> float:
    values = []
    for row in fake_ids:
        seq = _trim_ids(row)
        if len(seq) < 2:
            values.append(0.0)
            continue
        rep = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1]) / (len(seq) - 1)
        values.append(rep)
    return float(sum(values) / max(1, len(values)))


def ngram_diversity(fake_ids: torch.Tensor, n: int = 2) -> float:
    scores = []
    for row in fake_ids:
        seq = _trim_ids(row)
        if len(seq) < n:
            scores.append(0.0)
            continue
        grams = [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
        scores.append(len(set(grams)) / max(1, len(grams)))
    return float(sum(scores) / max(1, len(scores)))


def aa_frequency_kl(fake_ids: torch.Tensor, real_ids: torch.Tensor) -> float:
    def freq(t: torch.Tensor):
        c = Counter()
        total = 0
        for row in t:
            seq = _trim_ids(row)
            c.update(seq)
            total += len(seq)
        probs = []
        for idx in range(AA_START_IDX, int(fake_ids.max().item()) + 1 if fake_ids.numel() else AA_START_IDX):
            probs.append((c[idx] + 1e-6) / (total + 1e-6 * max(1, len(c))))
        if not probs:
            probs = [1.0]
        s = sum(probs)
        return [p / s for p in probs]
    p = freq(fake_ids)
    q = freq(real_ids)
    m = min(len(p), len(q))
    p = p[:m]
    q = q[:m]
    return float(sum(pi * math.log((pi + 1e-8) / (qi + 1e-8)) for pi, qi in zip(p, q)))


def predicted_lengths(fake_ids: torch.Tensor) -> torch.Tensor:
    lengths = []
    for row in fake_ids:
        length = 0
        for x in row.tolist():
            if x == EOS_IDX:
                break
            if x >= AA_START_IDX:
                length += 1
        lengths.append(length)
    return torch.tensor(lengths, dtype=torch.float32)


def length_mae(fake_ids: torch.Tensor, target_lengths: torch.Tensor) -> float:
    pred = predicted_lengths(fake_ids)
    return float(torch.abs(pred - target_lengths.float().cpu()).mean().item())


def eos_exact_rate(fake_ids: torch.Tensor, target_lengths: torch.Tensor) -> float:
    hits = []
    for row, tlen in zip(fake_ids, target_lengths.tolist()):
        row = row.tolist()
        eos_pos = row.index(EOS_IDX) if EOS_IDX in row else None
        hits.append(1.0 if eos_pos == int(tlen) else 0.0)
    return float(sum(hits) / max(1, len(hits)))
