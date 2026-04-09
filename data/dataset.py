from __future__ import annotations

from Bio import SeqIO
import torch
from torch.utils.data import Dataset

from config import MAX_AA_LEN
from utils import encode_sequence, clean_sequence


class ToxinAntitoxinDataset(Dataset):
    def __init__(self, toxin_fasta: str, antidote_fasta: str, toxin_embeddings_path: str):
        toxins = list(SeqIO.parse(toxin_fasta, 'fasta'))
        antidotes = list(SeqIO.parse(antidote_fasta, 'fasta'))
        if len(toxins) != len(antidotes):
            raise ValueError(f'Число токсинов и антитоксинов не совпадает: {len(toxins)} vs {len(antidotes)}')

        self.toxin_seqs = []
        self.antitoxin_seqs = []
        skipped_too_long = 0

        for t, a in zip(toxins, antidotes):
            toxin_seq = clean_sequence(str(t.seq))
            antitoxin_seq = clean_sequence(str(a.seq))
            if not toxin_seq or not antitoxin_seq:
                continue
            if len(antitoxin_seq) > MAX_AA_LEN:
                skipped_too_long += 1
                continue
            self.toxin_seqs.append(toxin_seq)
            self.antitoxin_seqs.append(antitoxin_seq)

        if not self.toxin_seqs:
            raise ValueError('После очистки и фильтрации не осталось валидных пар последовательностей.')

        cache = torch.load(toxin_embeddings_path, map_location='cpu')
        cached_sequences = cache.get('sequences')
        cached_embeddings = cache.get('embeddings')
        if cached_sequences is None or cached_embeddings is None:
            raise ValueError("Файл эмбеддингов должен содержать ключи 'sequences' и 'embeddings'.")

        seq_to_emb = {seq: emb for seq, emb in zip(cached_sequences, cached_embeddings)}
        self.toxin_embeddings = []
        filtered_toxin = []
        filtered_anti = []
        for toxin_seq, antitoxin_seq in zip(self.toxin_seqs, self.antitoxin_seqs):
            emb = seq_to_emb.get(toxin_seq)
            if emb is None:
                continue
            self.toxin_embeddings.append(torch.as_tensor(emb, dtype=torch.float32))
            filtered_toxin.append(toxin_seq)
            filtered_anti.append(antitoxin_seq)

        self.toxin_seqs = filtered_toxin
        self.antitoxin_seqs = filtered_anti
        self.toxin_embeddings = torch.stack(self.toxin_embeddings, dim=0)

        print(f'[dataset] Пропущено слишком длинных антитоксинов: {skipped_too_long} (лимит MAX_AA_LEN={MAX_AA_LEN})')
        print(f'[dataset] Валидных пар: {len(self.toxin_seqs)}')

    def __len__(self):
        return len(self.toxin_seqs)

    def __getitem__(self, idx):
        toxin_emb = self.toxin_embeddings[idx]
        decoder_input, target, aa_length = encode_sequence(self.antitoxin_seqs[idx])
        return (
            toxin_emb,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(aa_length, dtype=torch.long),
        )
