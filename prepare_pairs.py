from __future__ import annotations

import re
from Bio import SeqIO

from config import RAW_TOXIN_FASTA_PATH, RAW_ANTITOXIN_FASTA_PATH, TOXIN_FASTA_PATH, ANTITOXIN_FASTA_PATH


def main():
    toxins = {}
    antitoxins = {}
    t_re = re.compile(r'^T(\d+)$')
    at_re = re.compile(r'^AT(\d+)$')

    for record in SeqIO.parse(RAW_TOXIN_FASTA_PATH, 'fasta'):
        name = record.id.split()[0]
        m = t_re.match(name)
        if m:
            toxins[m.group(1)] = record

    for record in SeqIO.parse(RAW_ANTITOXIN_FASTA_PATH, 'fasta'):
        name = record.id.split()[0]
        m = at_re.match(name)
        if m:
            antitoxins[m.group(1)] = record

    paired_toxins, paired_antitoxins = [], []
    for number, t_record in toxins.items():
        a_record = antitoxins.get(number)
        if a_record is not None:
            paired_toxins.append(t_record)
            paired_antitoxins.append(a_record)

    print(f'Найдено пар: {len(paired_toxins)}')
    SeqIO.write(paired_toxins, TOXIN_FASTA_PATH, 'fasta')
    SeqIO.write(paired_antitoxins, ANTITOXIN_FASTA_PATH, 'fasta')
    print(f'Сохранено: {TOXIN_FASTA_PATH} и {ANTITOXIN_FASTA_PATH}')


if __name__ == '__main__':
    main()
