"""Датасет пар «токсин -> антидот» с предвычисленными ESM-эмбеддингами токсинов."""

from __future__ import annotations

from Bio import SeqIO
import torch
from torch.utils.data import Dataset

from config import MAX_AA_LEN
from utils import encode_sequence, clean_sequence


class ToxinAntitoxinDataset(Dataset):
    def __init__(self, toxin_fasta: str, antidote_fasta: str, toxin_embeddings_path: str):
        toxins = list(SeqIO.parse(toxin_fasta, "fasta"))
        antidotes = list(SeqIO.parse(antidote_fasta, "fasta"))

        if len(toxins) != len(antidotes):
            raise ValueError(
                f"Число токсинов и антитоксинов не совпадает: {len(toxins)} vs {len(antidotes)}"
            )

        self.toxins = []
        self.antitoxins = []
        skipped_too_long = 0

        for t, a in zip(toxins, antidotes):
            toxin_seq = clean_sequence(str(t.seq))
            antitoxin_seq = clean_sequence(str(a.seq))

            if len(toxin_seq) == 0 or len(antitoxin_seq) == 0:
                continue

            if len(antitoxin_seq) > MAX_AA_LEN:
                skipped_too_long += 1
                continue

            self.toxins.append(toxin_seq)
            self.antitoxins.append(antitoxin_seq)

        if len(self.toxins) == 0:
            raise ValueError("После очистки и фильтрации не осталось валидных пар последовательностей.")

        cache = torch.load(toxin_embeddings_path, map_location="cpu")
        cached_sequences = cache.get("sequences")
        cached_embeddings = cache.get("embeddings")

        if cached_sequences is None or cached_embeddings is None:
            raise ValueError("Файл эмбеддингов должен содержать ключи 'sequences' и 'embeddings'.")

        filtered_embeddings = []
        seq_to_emb = {seq: emb for seq, emb in zip(cached_sequences, cached_embeddings)}

        for toxin_seq in self.toxins:
            if toxin_seq not in seq_to_emb:
                raise ValueError(
                    "В кэше не найден эмбеддинг для одной из токсиновых последовательностей. "
                    "Пересоздайте toxin_embeddings.pt после обновления данных."
                )
            filtered_embeddings.append(seq_to_emb[toxin_seq])

        self.toxin_embeddings = torch.stack([
            torch.as_tensor(x, dtype=torch.float32) for x in filtered_embeddings
        ], dim=0)

        print(
            f"[dataset] Пропущено слишком длинных антитоксинов: {skipped_too_long} "
            f"(лимит MAX_AA_LEN={MAX_AA_LEN})"
        )

    def __len__(self):
        return len(self.toxins)

    def __getitem__(self, idx):
        toxin_emb = self.toxin_embeddings[idx]
        antitoxin_seq = self.antitoxins[idx]
        antitoxin_encoded, antitoxin_length = encode_sequence(antitoxin_seq)

        return (
            toxin_emb,
            torch.tensor(antitoxin_encoded, dtype=torch.long),
            torch.tensor(antitoxin_length, dtype=torch.long),
        )
