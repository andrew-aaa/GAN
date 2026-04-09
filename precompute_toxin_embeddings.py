"""Предвычисление и сохранение ESM-эмбеддингов токсинов для обучения."""

from __future__ import annotations

from Bio import SeqIO
import torch

from config import TOXIN_FASTA_PATH, ANTITOXIN_FASTA_PATH, TOXIN_EMBEDDINGS_PATH, MAX_AA_LEN
from esm_utils import get_esm_embedding
from utils import clean_sequence


def main():
    toxins = list(SeqIO.parse(TOXIN_FASTA_PATH, "fasta"))
    antidotes = list(SeqIO.parse(ANTITOXIN_FASTA_PATH, "fasta"))

    if len(toxins) == 0 or len(antidotes) == 0:
        raise ValueError("Парные FASTA пусты. Сначала запусти prepare_pairs.py.")

    if len(toxins) != len(antidotes):
        raise ValueError(
            f"Число токсинов и антитоксинов не совпадает: {len(toxins)} vs {len(antidotes)}"
        )

    sequences = []
    embeddings = []
    skipped_too_long = 0

    total = len(toxins)
    print(f"Предвычисление ESM-эмбеддингов для {total} пар токсин-антитоксин...")

    for idx, (toxin_record, antidote_record) in enumerate(zip(toxins, antidotes), start=1):
        toxin_seq = clean_sequence(str(toxin_record.seq))
        antidote_seq = clean_sequence(str(antidote_record.seq))

        if len(toxin_seq) == 0 or len(antidote_seq) == 0:
            continue

        if len(antidote_seq) > MAX_AA_LEN:
            skipped_too_long += 1
            continue

        embedding = get_esm_embedding(toxin_seq)
        sequences.append(toxin_seq)
        embeddings.append(embedding)

        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"  Обработано: {idx}/{total}")

    if len(sequences) == 0:
        raise ValueError("После очистки и фильтрации не осталось валидных токсиновых последовательностей.")

    embeddings_tensor = torch.stack(embeddings, dim=0)
    payload = {
        "sequences": sequences,
        "embeddings": embeddings_tensor,
    }
    torch.save(payload, TOXIN_EMBEDDINGS_PATH)

    print(f"Сохранено {len(sequences)} эмбеддингов в: {TOXIN_EMBEDDINGS_PATH}")
    print(f"Размер тензора: {tuple(embeddings_tensor.shape)}")
    if skipped_too_long > 0:
        print(f"Пропущено слишком длинных антитоксинов: {skipped_too_long}")


if __name__ == "__main__":
    main()
