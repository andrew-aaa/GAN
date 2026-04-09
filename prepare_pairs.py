"""Подготовка согласованных пар токсинов и антитоксинов из исходных FASTA-файлов."""

from __future__ import annotations

import re
from Bio import SeqIO

from config import (
    RAW_TOXIN_FASTA_PATH,
    RAW_ANTITOXIN_FASTA_PATH,
    TOXIN_FASTA_PATH,
    ANTITOXIN_FASTA_PATH,
)

TOXIN_RE = re.compile(r"^T(\d+)$")
ANTITOXIN_RE = re.compile(r"^AT(\d+)$")


def parse_toxin_number(name: str) -> str | None:
    match = TOXIN_RE.match(name)
    return match.group(1) if match else None


def parse_antitoxin_number(name: str) -> str | None:
    match = ANTITOXIN_RE.match(name)
    return match.group(1) if match else None


def main():
    toxins = {}
    antitoxins = {}

    skipped_toxin_ids = 0
    skipped_antitoxin_ids = 0

    for record in SeqIO.parse(RAW_TOXIN_FASTA_PATH, "fasta"):
        name = record.id.split()[0]
        number = parse_toxin_number(name)
        if number is None:
            skipped_toxin_ids += 1
            continue
        toxins[number] = record

    for record in SeqIO.parse(RAW_ANTITOXIN_FASTA_PATH, "fasta"):
        name = record.id.split()[0]
        number = parse_antitoxin_number(name)
        if number is None:
            skipped_antitoxin_ids += 1
            continue
        antitoxins[number] = record

    paired_numbers = sorted(set(toxins.keys()) & set(antitoxins.keys()), key=int)
    paired_toxins = [toxins[number] for number in paired_numbers]
    paired_antitoxins = [antitoxins[number] for number in paired_numbers]

    print(f"Найдено пар: {len(paired_toxins)}")
    print(f"Пропущено токсинов с нестандартным ID: {skipped_toxin_ids}")
    print(f"Пропущено антитоксинов с нестандартным ID: {skipped_antitoxin_ids}")

    SeqIO.write(paired_toxins, TOXIN_FASTA_PATH, "fasta")
    SeqIO.write(paired_antitoxins, ANTITOXIN_FASTA_PATH, "fasta")
    print(f"Сохранено: {TOXIN_FASTA_PATH}")
    print(f"Сохранено: {ANTITOXIN_FASTA_PATH}")


if __name__ == "__main__":
    main()
