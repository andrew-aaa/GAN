import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path

# Константы фильтрации (по твоим критериям)
MIN_UNIQUE_AA = 12
MAX_RUN = 4
MAX_SYMBOL_FRACTION = 0.20
MIN_LEN = 40
MAX_LEN = 250
MAX_IDENTITY = 0.95

def select_candidates(csv_path, fasta_out, filtered_csv_out):
    # 1. Загружаем результаты первичной валидации
    df = pd.read_csv(csv_path)
    total_generated = len(df)
    
    # 2. Применяем жесткие критерии отбора
    filtered_df = df[
        (df['valid_alphabet'] == True) &
        (df['exact_train_match'] == False) &
        (df['unique_aa'] >= MIN_UNIQUE_AA) &
        (df['max_run'] <= MAX_RUN) &
        (df['max_symbol_fraction'] <= MAX_SYMBOL_FRACTION) &
        (df['length'] >= MIN_LEN) &
        (df['length'] <= MAX_LEN) &
        (df['nearest_train_identity'] < MAX_IDENTITY)
    ].copy()

    # 3. Сохраняем отфильтрованную таблицу
    filtered_df.to_csv(filtered_csv_out, index=False)
    
    # 4. Формируем FASTA файл только с "золотыми" кандидатами
    records = []
    for _, row in filtered_df.iterrows():
        record = SeqRecord(
            Seq(row['sequence']),
            id=row['candidate_id'],
            description=f"toxin:{row['toxin_id']} ident:{row['nearest_train_identity']}"
        )
        records.append(record)
    
    SeqIO.write(records, fasta_out, "fasta")
    
    # Печать статистики для диплома
    print("=== Результаты фильтрации Step 1 ===")
    print(f"Всего сгенерировано: {total_generated}")
    print(f"Прошли фильтрацию:   {len(filtered_df)} ({len(filtered_df)/total_generated*100:.1f}%)")
    print(f"Отсеяно:             {total_generated - len(filtered_df)}")
    print(f"Файлы сохранены в папку validation/")

if __name__ == "__main__":
    csv_input = "validation/sequence_validation.csv"
    fasta_output = "validation/top_candidates.fasta"
    csv_output = "validation/top_candidates_metrics.csv"
    
    if Path(csv_input).exists():
        select_candidates(csv_input, fasta_output, csv_output)
    else:
        print(f"Ошибка: Файл {csv_input} не найден. Сначала запусти генерацию.")