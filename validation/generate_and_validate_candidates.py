from __future__ import annotations

"""
Первичная sequence-валидация сгенерированных антитоксинов.

Что делает скрипт:
1) Берёт обученный checkpoint генератора.
2) Выбирает первые N токсинов из data/toxins_paired.fasta.
3) Для каждого токсина генерирует K кандидатных антитоксинов.
4) Сохраняет кандидаты в validation/generated_candidates.fasta.
5) Считает первичные метрики валидности и novelty.
6) Сохраняет таблицу validation/sequence_validation.csv.

Запуск из корня проекта:
    python validation/generate_and_validate_candidates.py --num-toxins 10 --candidates-per-toxin 10

Для Colab:
    %cd /content/GAN
    !python validation/generate_and_validate_candidates.py --num-toxins 10 --candidates-per-toxin 10
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import torch

# Корень проекта: /content/GAN, если файл лежит в /content/GAN/validation/...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    AMINO_ACIDS,
    AA_START_IDX,
    ANTITOXIN_FASTA_PATH,
    DEVICE,
    GENERATION_TEMPERATURES,
    LATENT_DIM,
    MAX_AA_LEN,
    TOXIN_FASTA_PATH,
    TOXIN_EMBEDDINGS_PATH,
    VOCAB,
)

try:  # noqa: E402
    from config import GENERATOR_BEST_INFERENCE_PATH, EMA_BEST_INFERENCE_PATH
except ImportError:  # fallback для старой версии config.py
    GENERATOR_BEST_INFERENCE_PATH = None
    EMA_BEST_INFERENCE_PATH = None

try:  # noqa: E402
    from config import GENERATOR_BEST_PATH, EMA_BEST_PATH
except ImportError:
    GENERATOR_BEST_PATH = str(PROJECT_ROOT / "checkpoints" / "generator_best_val.pt")
    EMA_BEST_PATH = str(PROJECT_ROOT / "checkpoints" / "generator_ema_best_val.pt")

from models.generator import Generator  # noqa: E402
from esm_utils import get_esm_embedding  # noqa: E402

VALID_AA = set(AMINO_ACIDS)


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Простой FASTA-парсер без внешних зависимостей."""
    records: list[tuple[str, str]] = []
    cur_id: Optional[str] = None
    chunks: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(chunks).upper()))
                cur_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if cur_id is not None:
        records.append((cur_id, "".join(chunks).upper()))
    return records


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path, line_width: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec_id, seq in records:
            handle.write(f">{rec_id}\n")
            for i in range(0, len(seq), line_width):
                handle.write(seq[i : i + line_width] + "\n")


def choose_checkpoint() -> str:
    candidates = [
        EMA_BEST_INFERENCE_PATH,
        GENERATOR_BEST_INFERENCE_PATH,
        EMA_BEST_PATH,
        GENERATOR_BEST_PATH,
    ]
    for item in candidates:
        if item and Path(item).exists():
            return str(item)
    raise FileNotFoundError(
        "Не найден checkpoint генератора. Ожидаются файлы в checkpoints/: "
        "generator_ema_best_inference.pt / generator_best_inference.pt / generator_ema_best_val.pt / generator_best_val.pt"
    )


def load_embedding_cache(path: str | Path):
    """Загружает кэш ESM-эмбеддингов, если он есть.

    Поддерживает несколько возможных форматов:
    1) dict[id] -> tensor
    2) dict с ключами ids + embeddings/toxin_embeddings
    3) tensor/list в том же порядке, что и toxins_paired.fasta
    """
    p = Path(path)
    if not p.exists():
        print(f"[warning] Кэш эмбеддингов не найден: {p}. Буду считать ESM для выбранных токсинов на лету.")
        return None
    return torch.load(str(p), map_location="cpu")


def get_toxin_embedding(
    cache,
    toxin_id: str,
    toxin_seq: str,
    toxin_index: int,
) -> torch.Tensor:
    if cache is not None:
        if isinstance(cache, dict):
            # Прямой формат: {id: embedding}
            if toxin_id in cache and torch.is_tensor(cache[toxin_id]):
                return cache[toxin_id].detach().float()

            # Иногда id может быть сохранён с полным описанием или другим префиксом.
            for key, value in cache.items():
                if isinstance(key, str) and key.split()[0] == toxin_id and torch.is_tensor(value):
                    return value.detach().float()

            
            ids = cache.get("ids") or cache.get("toxin_ids") or cache.get("names")
            embs = cache.get("embeddings")
            if embs is None:
                embs = cache.get("toxin_embeddings")
            if embs is None:
                embs = cache.get("embs")

            if ids is not None and embs is not None and toxin_id in ids:
                idx = list(ids).index(toxin_id)
                return torch.as_tensor(embs[idx]).detach().float()

        if torch.is_tensor(cache):
            return cache[toxin_index].detach().float()

        if isinstance(cache, (list, tuple)):
            return torch.as_tensor(cache[toxin_index]).detach().float()

    # fallback: медленнее, но надёжно
    return get_esm_embedding(toxin_seq).detach().float()


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


def max_symbol_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    counts = Counter(seq)
    return max(counts.values()) / len(seq)


def ngram_diversity(seq: str, n: int = 3) -> float:
    if len(seq) < n:
        return 0.0
    grams = [seq[i : i + n] for i in range(len(seq) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def kmer_set(seq: str, k: int = 3) -> set[str]:
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def pairwise_identity_biopython(query: str, target: str) -> Optional[tuple[float, float, float]]:
    """Возвращает (identity, coverage_query, score) через Biopython PairwiseAligner.

    identity считается по колонкам выравнивания с учётом gap-колонок.
    coverage_query — доля query, попавшая в aligned-блоки без gap в query.
    Если Biopython недоступен, возвращает None.
    """
    try:
        from Bio import Align  # type: ignore
    except Exception:
        return None

    try:
        aligner = Align.PairwiseAligner(scoring="blastp")
        aligner.mode = "global"
        alignment = aligner.align(target, query)[0]
        coords = alignment.coordinates

        matches = 0
        aligned_cols = 0
        query_covered = 0

        for i in range(coords.shape[1] - 1):
            t0, t1 = int(coords[0, i]), int(coords[0, i + 1])
            q0, q1 = int(coords[1, i]), int(coords[1, i + 1])
            dt = t1 - t0
            dq = q1 - q0

            if dt > 0 and dq > 0:
                block_t = target[t0:t1]
                block_q = query[q0:q1]
                block_len = min(len(block_t), len(block_q))
                matches += sum(1 for a, b in zip(block_t[:block_len], block_q[:block_len]) if a == b)
                aligned_cols += max(dt, dq)
                query_covered += dq
            else:
                aligned_cols += max(dt, dq)
                query_covered += dq

        identity = matches / max(1, aligned_cols)
        coverage = query_covered / max(1, len(query))
        return float(identity), float(coverage), float(alignment.score)
    except Exception:
        return None


def nearest_train_similarity(
    seq: str,
    train_records: list[tuple[str, str]],
    alignment_top_k: int = 25,
    k: int = 3,
) -> dict[str, object]:
    """Быстрый поиск ближайшего train-антитоксина.

    Сначала грубо отбирает top-K по k-mer Jaccard, затем, если установлен Biopython,
    уточняет identity через PairwiseAligner. Это не замена BLAST/MMseqs2, а дешёвая
    локальная первичная проверка novelty.
    """
    q_kmers = kmer_set(seq, k=k)
    rough = []
    for rec_id, train_seq in train_records:
        sim = jaccard(q_kmers, kmer_set(train_seq, k=k))
        rough.append((sim, rec_id, train_seq))
    rough.sort(reverse=True, key=lambda x: x[0])
    top = rough[: max(1, alignment_top_k)]

    best = {
        "nearest_train_id": top[0][1] if top else "",
        "nearest_train_kmer_jaccard": float(top[0][0]) if top else 0.0,
        "nearest_train_identity": math.nan,
        "nearest_train_coverage": math.nan,
        "nearest_train_alignment_score": math.nan,
        "nearest_method": "kmer_jaccard_only",
    }

    best_identity = -1.0
    for _, rec_id, train_seq in top:
        result = pairwise_identity_biopython(seq, train_seq)
        if result is None:
            continue
        identity, coverage, score = result
        if identity > best_identity:
            best_identity = identity
            best.update(
                {
                    "nearest_train_id": rec_id,
                    "nearest_train_identity": identity,
                    "nearest_train_coverage": coverage,
                    "nearest_train_alignment_score": score,
                    "nearest_method": "biopython_pairwisealigner_blastp_global",
                }
            )

    return best


def fixed_length_decode_from_probs(probs_2d: torch.Tensor, target_length: int) -> str:
    """Декодирует ровно target_length канонических аминокислот.

    Если argmax даёт служебный токен PAD/BOS/EOS, берём наиболее вероятную
    каноническую аминокислоту на этой позиции.
    """
    target_length = max(1, min(int(target_length), int(MAX_AA_LEN)))
    probs_2d = probs_2d.detach().cpu()
    chars: list[str] = []

    for pos in range(min(target_length, probs_2d.size(0))):
        row = probs_2d[pos]
        idx = int(row.argmax().item())
        if idx >= AA_START_IDX and idx < len(VOCAB) and VOCAB[idx] in VALID_AA:
            chars.append(VOCAB[idx])
        else:
            aa_idx = int(row[AA_START_IDX:].argmax().item()) + AA_START_IDX
            chars.append(VOCAB[aa_idx])

    if len(chars) < target_length:
        filler = Counter(chars).most_common(1)[0][0] if chars else "A"
        chars.extend([filler] * (target_length - len(chars)))

    return "".join(chars[:target_length])


def generate_candidates_for_toxin(
    generator: Generator,
    toxin_emb: torch.Tensor,
    candidates_per_toxin: int,
    attempts_multiplier: int,
    temperatures: tuple[float, ...],
) -> tuple[int, list[dict[str, object]]]:
    toxin_emb = toxin_emb.unsqueeze(0).to(DEVICE).float()
    generated: dict[str, dict[str, object]] = {}

    with torch.no_grad():
        predicted_length = int(generator.predict_lengths(toxin_emb).item())
        predicted_length = max(1, min(predicted_length, int(MAX_AA_LEN)))
        target_lengths = torch.tensor([predicted_length], device=DEVICE)

        total_attempts_per_temp = max(1, candidates_per_toxin * attempts_multiplier)
        for temperature in temperatures:
            for _ in range(total_attempts_per_temp):
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                sample_out = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=target_lengths,
                    temperature=float(temperature),
                    hard=False,
                )
                probs = sample_out[0] if isinstance(sample_out, tuple) else sample_out
                probs_2d = probs.squeeze(0)
                seq = fixed_length_decode_from_probs(probs_2d, predicted_length)

                if seq not in generated:
                    uniq = len(set(seq))
                    run = max_run_length(seq)
                    max_frac = max_symbol_fraction(seq)
                    score = 3.0 * abs(len(seq) - predicted_length) + 2.0 * run + 20.0 * max_frac - 0.5 * uniq
                    generated[seq] = {
                        "sequence": seq,
                        "temperature": float(temperature),
                        "candidate_score": float(score),
                    }

    ranked = sorted(generated.values(), key=lambda x: float(x["candidate_score"]))
    return predicted_length, ranked[:candidates_per_toxin]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and validate candidate antitoxin sequences.")
    parser.add_argument("--num-toxins", type=int, default=10, help="Сколько токсинов взять из toxins_paired.fasta.")
    parser.add_argument("--start-index", type=int, default=0, help="С какого индекса токсина начать.")
    parser.add_argument("--candidates-per-toxin", type=int, default=10, help="Сколько кандидатов оставить на токсин.")
    parser.add_argument("--attempts-multiplier", type=int, default=4, help="Попыток генерации на один кандидат и температуру.")
    parser.add_argument("--temperatures", type=float, nargs="*", default=None, help="Температуры генерации. По умолчанию из config.py.")
    parser.add_argument("--alignment-top-k", type=int, default=25, help="Сколько ближайших train-последовательностей уточнять PairwiseAligner-ом.")
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "validation"), help="Куда сохранить FASTA/CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Seed для torch.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_out = out_dir / "generated_candidates.fasta"
    csv_out = out_dir / "sequence_validation.csv"
    summary_out = out_dir / "sequence_validation_summary.json"

    toxin_records = parse_fasta(TOXIN_FASTA_PATH)
    antitoxin_records = parse_fasta(ANTITOXIN_FASTA_PATH)
    antitoxin_train_set = {seq for _, seq in antitoxin_records}

    selected = toxin_records[args.start_index : args.start_index + args.num_toxins]
    if not selected:
        raise ValueError("Не выбрано ни одного токсина. Проверь --start-index/--num-toxins и FASTA-файл.")

    ckpt = choose_checkpoint()
    print(f"[checkpoint] {ckpt}")
    print(f"[device] {DEVICE}")
    print(f"[toxins] selected={len(selected)} from {TOXIN_FASTA_PATH}")
    print(f"[train antitoxins] {len(antitoxin_records)} from {ANTITOXIN_FASTA_PATH}")

    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    generator.eval()

    cache = load_embedding_cache(TOXIN_EMBEDDINGS_PATH)
    temperatures = tuple(args.temperatures) if args.temperatures else tuple(GENERATION_TEMPERATURES)

    fasta_records: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []

    for local_idx, (toxin_id, toxin_seq) in enumerate(selected):
        global_idx = args.start_index + local_idx
        print(f"[generate] toxin {local_idx + 1}/{len(selected)} id={toxin_id}")
        toxin_emb = get_toxin_embedding(cache, toxin_id, toxin_seq, global_idx)
        predicted_length, candidates = generate_candidates_for_toxin(
            generator=generator,
            toxin_emb=toxin_emb,
            candidates_per_toxin=args.candidates_per_toxin,
            attempts_multiplier=args.attempts_multiplier,
            temperatures=temperatures,
        )

        for rank, item in enumerate(candidates, start=1):
            seq = str(item["sequence"])
            candidate_id = f"gen_t{global_idx:04d}_r{rank:02d}"
            valid_alphabet = all(ch in VALID_AA for ch in seq)
            exact_train_match = seq in antitoxin_train_set
            nearest = nearest_train_similarity(
                seq,
                antitoxin_records,
                alignment_top_k=args.alignment_top_k,
                k=3,
            )

            fasta_header = (
                f"{candidate_id}|toxin_id={toxin_id}|rank={rank}|pred_len={predicted_length}|"
                f"temp={item['temperature']}"
            )
            fasta_records.append((fasta_header, seq))

            row = {
                "candidate_id": candidate_id,
                "toxin_id": toxin_id,
                "toxin_index": global_idx,
                "rank": rank,
                "sequence": seq,
                "length": len(seq),
                "predicted_length": predicted_length,
                "length_abs_error": abs(len(seq) - predicted_length),
                "temperature": item["temperature"],
                "candidate_score": item["candidate_score"],
                "valid_alphabet": valid_alphabet,
                "unique_aa": len(set(seq)),
                "max_run": max_run_length(seq),
                "max_symbol_fraction": max_symbol_fraction(seq),
                "ngram2_diversity": ngram_diversity(seq, n=2),
                "ngram3_diversity": ngram_diversity(seq, n=3),
                "exact_train_match": exact_train_match,
                **nearest,
            }
            rows.append(row)

    write_fasta(fasta_records, fasta_out)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accepted = [
        r for r in rows
        if bool(r["valid_alphabet"])
        and not bool(r["exact_train_match"])
        and int(r["unique_aa"]) >= 12
        and int(r["max_run"]) <= 4
        and float(r["max_symbol_fraction"]) <= 0.20
        and 40 <= int(r["length"]) <= int(MAX_AA_LEN)
        and (math.isnan(float(r["nearest_train_identity"])) or float(r["nearest_train_identity"]) < 0.95)
    ]

    summary = {
        "checkpoint": ckpt,
        "num_toxins": len(selected),
        "candidates_total": len(rows),
        "accepted_by_basic_filters": len(accepted),
        "generated_fasta": str(fasta_out),
        "validation_csv": str(csv_out),
        "filters": {
            "valid_alphabet": True,
            "exact_train_match": False,
            "unique_aa_min": 12,
            "max_run_max": 4,
            "max_symbol_fraction_max": 0.20,
            "length_range": [40, int(MAX_AA_LEN)],
            "nearest_train_identity_max": 0.95,
        },
    }
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n[done]")
    print(f"FASTA:   {fasta_out}")
    print(f"CSV:     {csv_out}")
    print(f"SUMMARY: {summary_out}")
    print(f"Accepted by basic filters: {len(accepted)}/{len(rows)}")
    if rows and math.isnan(float(rows[0]["nearest_train_identity"])):
        print("[note] Biopython PairwiseAligner не использован. Для identity установи: pip install biopython")


if __name__ == "__main__":
    main()
