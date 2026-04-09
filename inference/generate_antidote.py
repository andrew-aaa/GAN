"""Генерация кандидатного антидота по заданной последовательности токсина."""

from __future__ import annotations

from pathlib import Path

import torch

from config import (
    DEVICE,
    LATENT_DIM,
    EMA_MODEL_SAVE_PATH,
    BEST_EMA_PROXY_MODEL_SAVE_PATH,
    BEST_PROXY_MODEL_SAVE_PATH,
    MODEL_SAVE_PATH,
)
from esm_utils import get_esm_embedding
from models.generator import Generator
from utils import decode_sequence


DEFAULT_TOXIN = (
    "VLKLNLKKSFQKDFDKLLLNGFDDSVLNEVILTLRKKEPLDPQFQDHALKGKWKPFRECHIKPDV"
    "LLVYLVKDDELILLRLGSHSELF"
)


def candidate_score(seq: str, target_length: int) -> tuple[float, int, int]:
    if not seq:
        return (10_000.0, 0, 0)
    uniq = len(set(seq))
    repeats = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    length_pen = abs(len(seq) - target_length)
    score = 2.5 * length_pen + 1.5 * repeats - 1.0 * uniq
    return (score, uniq, repeats)


def pick_checkpoint() -> str:
    for path in (
        BEST_EMA_PROXY_MODEL_SAVE_PATH,
        EMA_MODEL_SAVE_PATH,
        BEST_PROXY_MODEL_SAVE_PATH,
        MODEL_SAVE_PATH,
    ):
        if Path(path).exists():
            return path
    raise FileNotFoundError("Не найден ни один чекпоинт генератора.")


def main(toxin_seq: str = DEFAULT_TOXIN):
    checkpoint_path = pick_checkpoint()

    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    generator.eval()

    toxin_emb = get_esm_embedding(toxin_seq).unsqueeze(0).to(DEVICE).float()

    best = None
    with torch.no_grad():
        predicted_length = int(generator.predict_lengths(toxin_emb).item())
        # По аудиту: температура должна масштабировать логиты, а не tau при hard=True.
        for sampling_temperature in (1.2, 1.0, 0.9, 0.8, 0.7):
            for _ in range(8):
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                antidote_probs, _ = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=torch.tensor([predicted_length], device=DEVICE),
                    sampling_temperature=sampling_temperature,
                    gumbel_tau=1.0,
                    hard=True,
                )
                antidote_idx = antidote_probs.argmax(dim=-1).squeeze(0).cpu().numpy()
                antidote_seq = decode_sequence(antidote_idx)
                score, uniq, repeats = candidate_score(antidote_seq, predicted_length)
                item = (score, antidote_seq, uniq, repeats, sampling_temperature)
                if best is None or item[0] < best[0]:
                    best = item

    best_score, antidote_seq, uniq, repeats, sampling_temperature = best

    print("Предсказанный антидот для токсина:")
    print(f"  Чекпойнт: {checkpoint_path}")
    print(f"  Предсказанная длина: {predicted_length}")
    print(f"  Лучшая температура семплирования: {sampling_temperature}")
    print(f"  Уникальных аминокислот: {uniq}")
    print(f"  Повторов подряд: {repeats}")
    print(f"  Score: {best_score:.3f}")
    print(f"  Последовательность: {antidote_seq}")


if __name__ == "__main__":
    main()
