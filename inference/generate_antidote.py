from __future__ import annotations

from pathlib import Path

import torch

from config import DEVICE, LATENT_DIM, GENERATOR_BEST_PATH, EMA_BEST_PATH
from esm_utils import get_esm_embedding
from models.generator import Generator
from utils import decode_sequence


def candidate_score(seq: str, target_length: int) -> tuple[float, int, int]:
    if not seq:
        return (10_000.0, 0, 0)
    uniq = len(set(seq))
    repeats = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    length_pen = abs(len(seq) - target_length)
    score = 2.0 * length_pen + 1.5 * repeats - 1.0 * uniq
    return (score, uniq, repeats)


def main():
    ckpt = EMA_BEST_PATH if Path(EMA_BEST_PATH).exists() else GENERATOR_BEST_PATH
    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    generator.eval()

    toxin_seq = 'VLKLNLKKSFQKDFDKLLLNGFDDSVLNEVILTLRKKEPLDPQFQDHALKGKWKPFRECHIKPDVLLVYLVKDDELILLRLGSHSELF'
    toxin_emb = get_esm_embedding(toxin_seq).unsqueeze(0).to(DEVICE).float()

    best = None
    with torch.no_grad():
        predicted_length = int(generator.predict_lengths(toxin_emb).item())
        for temperature in (1.2, 1.0, 0.9, 0.8):
            for _ in range(8):
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                probs, _ = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=torch.tensor([predicted_length], device=DEVICE),
                    temperature=temperature,
                    hard=True,
                )
                ids = probs.argmax(dim=-1).squeeze(0).cpu().numpy()
                seq = decode_sequence(ids)
                score, uniq, repeats = candidate_score(seq, predicted_length)
                item = (score, seq, uniq, repeats, temperature)
                if best is None or item[0] < best[0]:
                    best = item

    score, seq, uniq, repeats, temp = best
    print('Предсказанный антидот для токсина:')
    print(f'  Чекпойнт: {ckpt}')
    print(f'  Предсказанная длина: {predicted_length}')
    print(f'  Лучшая температура: {temp}')
    print(f'  Уникальных аминокислот: {uniq}')
    print(f'  Повторов подряд: {repeats}')
    print(f'  Последовательность: {seq}')


if __name__ == '__main__':
    main()
