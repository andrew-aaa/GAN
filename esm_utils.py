from __future__ import annotations

from functools import lru_cache

import torch

from config import DEVICE

_model = None
_alphabet = None
_batch_converter = None


def _load_esm():
    global _model, _alphabet, _batch_converter
    if _model is not None:
        return _model, _alphabet, _batch_converter
    try:
        import esm
    except ImportError as exc:
        raise ImportError("Не найден пакет 'esm'. Установи fair-esm: pip install fair-esm") from exc
    _model, _alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    _batch_converter = _alphabet.get_batch_converter()
    _model.eval()
    _model = _model.to(DEVICE)
    return _model, _alphabet, _batch_converter


@torch.no_grad()
@lru_cache(maxsize=4096)
def get_esm_embedding(sequence: str) -> torch.Tensor:
    model, _, batch_converter = _load_esm()
    data = [('protein', sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)
    results = model(tokens, repr_layers=[6], return_contacts=False)
    reps = results['representations'][6]
    if reps.size(1) > 2:
        emb = reps[:, 1:-1, :].mean(dim=1)
    else:
        emb = reps.mean(dim=1)
    return emb.squeeze(0).detach().cpu()
