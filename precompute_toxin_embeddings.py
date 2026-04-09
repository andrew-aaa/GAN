from __future__ import annotations

from Bio import SeqIO
import torch

from config import TOXIN_FASTA_PATH, ANTITOXIN_FASTA_PATH, TOXIN_EMBEDDINGS_PATH, MAX_AA_LEN
from esm_utils import get_esm_embedding
from utils import clean_sequence


def main():
    toxins = list(SeqIO.parse(TOXIN_FASTA_PATH, 'fasta'))
    antidotes = list(SeqIO.parse(ANTITOXIN_FASTA_PATH, 'fasta'))
    if len(toxins) != len(antidotes):
        raise ValueError(f'Число токсинов и антитоксинов не совпадает: {len(toxins)} vs {len(antidotes)}')

    sequences, embeddings = [], []
    skipped = 0
    for i, (t, a) in enumerate(zip(toxins, antidotes), start=1):
        toxin_seq = clean_sequence(str(t.seq))
        antidote_seq = clean_sequence(str(a.seq))
        if not toxin_seq or not antidote_seq:
            continue
        if len(antidote_seq) > MAX_AA_LEN:
            skipped += 1
            continue
        sequences.append(toxin_seq)
        embeddings.append(get_esm_embedding(toxin_seq))
        if i == 1 or i % 25 == 0 or i == len(toxins):
            print(f'  Обработано: {i}/{len(toxins)}')

    payload = {'sequences': sequences, 'embeddings': torch.stack(embeddings, dim=0)}
    torch.save(payload, TOXIN_EMBEDDINGS_PATH)
    print(f'Сохранено {len(sequences)} эмбеддингов в: {TOXIN_EMBEDDINGS_PATH}')
    print(f'Пропущено слишком длинных антитоксинов: {skipped}')


if __name__ == '__main__':
    main()
