from __future__ import annotations

from pathlib import Path
import sys
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import DEVICE, LATENT_DIM, GENERATOR_BEST_PATH, EMA_BEST_PATH
try:
    from config import GENERATOR_BEST_INFERENCE_PATH, EMA_BEST_INFERENCE_PATH
except ImportError:
    GENERATOR_BEST_INFERENCE_PATH = GENERATOR_BEST_PATH
    EMA_BEST_INFERENCE_PATH = EMA_BEST_PATH
try:
    from config import GENERATION_TEMPERATURES, NUM_GENERATION_ATTEMPTS
except ImportError:
    GENERATION_TEMPERATURES = (1.2, 1.0, 0.9, 0.8)
    NUM_GENERATION_ATTEMPTS = 16

from esm_utils import get_esm_embedding
from models.generator import Generator
from utils import decode_sequence


def choose_checkpoint() -> str:
    candidates = [
        EMA_BEST_INFERENCE_PATH,
        EMA_BEST_PATH,
        GENERATOR_BEST_INFERENCE_PATH,
        GENERATOR_BEST_PATH,
    ]
    for path in candidates:
        if path and Path(path).exists():
            return str(path)
    raise FileNotFoundError("Не найден ни один checkpoint генератора в папке checkpoints/.")


def max_run_length(seq: str) -> int:
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def candidate_score(seq: str, target_length: int) -> tuple[float, int, int, float]:
    if not seq:
        return (10_000.0, 0, 0, 1.0)
    uniq = len(set(seq))
    run = max_run_length(seq)
    length_pen = abs(len(seq) - target_length)
    counts = Counter(seq)
    max_freq = max(counts.values()) / max(1, len(seq))

    # Чем меньше score, тем лучше: штрафуем неверную длину, повторы и доминирование одного символа,
    # немного поощряем аминокислотное разнообразие.
    score = 3.0 * length_pen + 2.0 * run + 20.0 * max_freq - 0.5 * uniq
    return (score, uniq, run, max_freq)


def main():
    ckpt = choose_checkpoint()
    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    generator.eval()

    toxin_seq = "VLKLNLKKSFQKDFDKLLLNGFDDSVLNEVILTLRKKEPLDPQFQDHALKGKWKPFRECHIKPDVLLVYLVKDDELILLRLGSHSELF"
    toxin_emb = get_esm_embedding(toxin_seq).unsqueeze(0).to(DEVICE).float()

    best = None
    with torch.no_grad():
        predicted_length = int(generator.predict_lengths(toxin_emb).item())
        target_lengths = torch.tensor([predicted_length], device=DEVICE)

        for temperature in GENERATION_TEMPERATURES:
            for _ in range(NUM_GENERATION_ATTEMPTS):
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                probs, _ = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=target_lengths,
                    temperature=float(temperature),
                    hard=True,
                )
                ids = probs.argmax(dim=-1).squeeze(0).cpu().tolist()
                seq = decode_sequence(ids)
                score, uniq, run, max_freq = candidate_score(seq, predicted_length)
                item = (score, seq, uniq, run, max_freq, temperature)
                if best is None or item[0] < best[0]:
                    best = item

    score, seq, uniq, run, max_freq, temp = best
    print("Предсказанный антидот для токсина:")
    print(f"  Чекпойнт: {ckpt}")
    print(f"  Предсказанная длина: {predicted_length}")
    print(f"  Фактическая длина: {len(seq)}")
    print(f"  Лучшая температура: {temp}")
    print(f"  Уникальных аминокислот: {uniq}")
    print(f"  Максимальный повтор подряд: {run}")
    print(f"  Максимальная доля одного символа: {max_freq:.3f}")
    print(f"  Последовательность: {seq}")


if __name__ == "__main__":
    main()
