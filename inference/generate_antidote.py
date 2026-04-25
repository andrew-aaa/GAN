from __future__ import annotations

from pathlib import Path
import sys
import argparse
from collections import Counter
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    DEVICE,
    LATENT_DIM,
    GENERATOR_BEST_PATH,
    EMA_BEST_PATH,
    AMINO_ACIDS,
    VOCAB,
    AA_START_IDX,
    MAX_AA_LEN,
)

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


DEFAULT_TOXIN_FASTA = PROJECT_ROOT / "data" / "target_toxin.fasta"
DEFAULT_OUTPUT_FASTA = PROJECT_ROOT / "outputs" / "generated_antidote.fasta"


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """
    Reads one or more sequences from a FASTA file.

    Returns:
        List of (record_id, sequence) tuples.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {path}\n"
            f"Create it, for example: {DEFAULT_TOXIN_FASTA}"
        )

    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_seq: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq).upper()))

                header = line[1:].strip()
                current_id = header.split()[0] if header else f"record_{len(records) + 1}"
                current_seq = []
            else:
                current_seq.append(line.replace(" ", "").upper())

    if current_id is not None:
        records.append((current_id, "".join(current_seq).upper()))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def choose_fasta_record(records: list[tuple[str, str]], record_id: str | None = None) -> tuple[str, str]:
    if record_id is None:
        return records[0]

    for rid, seq in records:
        if rid == record_id:
            return rid, seq

    available = ", ".join(rid for rid, _ in records[:10])
    raise ValueError(
        f"Record id '{record_id}' not found in FASTA. "
        f"Available examples: {available}"
    )


def validate_protein_sequence(seq: str, name: str = "sequence") -> None:
    allowed = set(AMINO_ACIDS)
    invalid = sorted(set(seq) - allowed)
    if invalid:
        raise ValueError(
            f"{name} contains non-canonical amino acid symbols: {invalid}. "
            f"Allowed alphabet: {AMINO_ACIDS}"
        )
    if not seq:
        raise ValueError(f"{name} is empty.")


def choose_checkpoint(user_checkpoint: str | None = None) -> str:
    if user_checkpoint:
        path = Path(user_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    candidates = [
        EMA_BEST_INFERENCE_PATH,
        EMA_BEST_PATH,
        GENERATOR_BEST_INFERENCE_PATH,
        GENERATOR_BEST_PATH,
    ]
    for path in candidates:
        if path and Path(path).exists():
            return str(path)

    raise FileNotFoundError("No generator checkpoint found in checkpoints/.")


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


def fixed_length_decode_from_probs(probs: torch.Tensor, target_length: int) -> str:
    """
    Decodes exactly target_length amino acids.

    If argmax gives a special token (PAD/BOS/EOS) inside the target region,
    it is replaced with the most probable canonical amino acid at this position.
    This keeps the final output length equal to predicted_length.
    """
    if probs.dim() == 3:
        probs = probs.squeeze(0)

    target_length = int(max(1, min(target_length, MAX_AA_LEN, probs.size(0))))
    ids = probs.argmax(dim=-1).detach().cpu().tolist()

    seq: list[str] = []
    aa_chars = list(AMINO_ACIDS)

    for pos in range(target_length):
        token = VOCAB[ids[pos]]

        if token in AMINO_ACIDS:
            seq.append(token)
            continue

        # Replace service token with the most probable canonical amino acid.
        aa_probs = probs[pos, AA_START_IDX : AA_START_IDX + len(AMINO_ACIDS)]
        aa_idx = int(torch.argmax(aa_probs).item())
        seq.append(aa_chars[aa_idx])

    return "".join(seq)


def candidate_score(seq: str, target_length: int) -> tuple[float, int, int, float]:
    if not seq:
        return (10_000.0, 0, 0, 1.0)

    uniq = len(set(seq))
    run = max_run_length(seq)
    length_pen = abs(len(seq) - target_length)
    counts = Counter(seq)
    max_freq = max(counts.values()) / max(1, len(seq))

    # Lower is better.
    # Penalize wrong length, long repeats and single-symbol dominance;
    # reward amino-acid diversity.
    score = 3.0 * length_pen + 2.0 * run + 20.0 * max_freq - 0.5 * uniq
    return (score, uniq, run, max_freq)


def write_fasta(path: str | Path, record_id: str, sequence: str, line_width: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{record_id}\n")
        for start in range(0, len(sequence), line_width):
            handle.write(sequence[start : start + line_width] + "\n")


def parse_temperatures(value: str | None) -> tuple[float, ...]:
    if value is None:
        return tuple(float(x) for x in GENERATION_TEMPERATURES)

    temps = []
    for item in value.split(","):
        item = item.strip()
        if item:
            temps.append(float(item))

    if not temps:
        raise ValueError("At least one temperature must be provided.")

    return tuple(temps)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one antitoxin candidate for a target toxin loaded from a FASTA file."
    )
    parser.add_argument(
        "--toxin-file",
        default=str(DEFAULT_TOXIN_FASTA),
        help=f"Path to FASTA file with target toxin. Default: {DEFAULT_TOXIN_FASTA}",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Optional FASTA record id. If not set, the first sequence is used.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to generator checkpoint. If not set, the best available checkpoint is used.",
    )
    parser.add_argument(
        "--output-fasta",
        default=str(DEFAULT_OUTPUT_FASTA),
        help=f"Where to save the generated antitoxin FASTA. Default: {DEFAULT_OUTPUT_FASTA}",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(NUM_GENERATION_ATTEMPTS),
        help=f"Number of attempts per temperature. Default: {NUM_GENERATION_ATTEMPTS}",
    )
    parser.add_argument(
        "--temperatures",
        default=None,
        help=(
            "Comma-separated temperatures, for example: 1.2,1.0,0.9. "
            f"Default from config: {GENERATION_TEMPERATURES}"
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    temperatures = parse_temperatures(args.temperatures)
    toxin_records = read_fasta(args.toxin_file)
    toxin_id, toxin_seq = choose_fasta_record(toxin_records, args.record_id)
    validate_protein_sequence(toxin_seq, name=f"toxin '{toxin_id}'")

    ckpt = choose_checkpoint(args.checkpoint)

    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    generator.eval()

    toxin_emb = get_esm_embedding(toxin_seq).unsqueeze(0).to(DEVICE).float()

    best = None
    with torch.no_grad():
        predicted_length = int(generator.predict_lengths(toxin_emb).item())
        predicted_length = max(1, min(predicted_length, MAX_AA_LEN))
        target_lengths = torch.tensor([predicted_length], device=DEVICE)

        for temperature in temperatures:
            for _ in range(max(1, args.attempts)):
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                probs, _ = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=target_lengths,
                    temperature=float(temperature),
                    hard=True,
                )
                seq = fixed_length_decode_from_probs(probs, predicted_length)
                score, uniq, run, max_freq = candidate_score(seq, predicted_length)
                item = (score, seq, uniq, run, max_freq, temperature)
                if best is None or item[0] < best[0]:
                    best = item

    if best is None:
        raise RuntimeError("Generation failed: no candidate was produced.")

    score, seq, uniq, run, max_freq, temp = best
    out_id = f"generated_antitoxin_for_{toxin_id}_len{predicted_length}"
    write_fasta(args.output_fasta, out_id, seq)

    print("Предсказанный антидот для токсина:")
    print(f"  Файл токсина: {args.toxin_file}")
    print(f"  ID токсина: {toxin_id}")
    print(f"  Длина токсина: {len(toxin_seq)}")
    print(f"  Чекпойнт: {ckpt}")
    print(f"  Предсказанная длина: {predicted_length}")
    print(f"  Фактическая длина: {len(seq)}")
    print(f"  Лучшая температура: {temp}")
    print(f"  Уникальных аминокислот: {uniq}")
    print(f"  Максимальный повтор подряд: {run}")
    print(f"  Максимальная доля одного символа: {max_freq:.3f}")
    print(f"  Candidate score: {score:.4f}")
    print(f"  FASTA сохранён: {args.output_fasta}")
    print(f"  Последовательность: {seq}")


if __name__ == "__main__":
    main()
