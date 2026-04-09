"""Утилиты для получения фиксированного ESM-представления белковой последовательности."""

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
        raise ImportError(
            "Не найден пакет 'esm'. Установите зависимости из requirements.txt "
            "(в частности fair-esm), затем повторите запуск."
        ) from exc

    _model, _alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    _batch_converter = _alphabet.get_batch_converter()

    _model.eval()
    _model = _model.to(DEVICE)
    return _model, _alphabet, _batch_converter


@torch.no_grad()
@lru_cache(maxsize=4096)
def get_esm_embedding(sequence: str) -> torch.Tensor:
    """Возвращает усреднённый ESM-эмбеддинг белка размера ESM_DIM."""
    model, _, batch_converter = _load_esm()

    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    results = model(tokens, repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]

    if token_representations.size(1) > 2:
        embedding = token_representations[:, 1:-1, :].mean(dim=1)
    else:
        embedding = token_representations.mean(dim=1)

    return embedding.squeeze(0).detach().cpu()
