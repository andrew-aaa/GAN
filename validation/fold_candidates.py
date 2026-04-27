# Сохраните это в /content/GAN/validation/fold_candidates.py
import pandas as pd
import torch
import esm
import os
from pathlib import Path

def main():
    # Настройки
    csv_path = Path("/content/GAN/validation/sequence_validation.csv")
    output_dir = Path("/content/GAN/validation/structures")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Выбираем лучших кандидатов
    df = pd.read_csv(csv_path)
    # Фильтруем: берем тех, кто прошел базовые фильтры и имеет лучший score
    # (чем меньше score, тем лучше)
    top_candidates = df.sort_values("candidate_score").head(5)
    
    print(f"Загрузка ESMFold...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    
    # Опционально для экономии памяти на длинных белках
    # model.set_chunk_size(128)

    print(f"Начинаю предсказание для {len(top_candidates)} структур...")
    
    for i, row in top_candidates.iterrows():
        name = row['candidate_id']
        seq = row['sequence']
        
        print(f"Фолдинг {name} (длина {len(seq)})...")
        
        with torch.no_grad():
            output = model.infer_pdb(seq)
            
        pdb_path = output_dir / f"{name}.pdb"
        with open(pdb_path, "w") as f:
            f.write(output)
            
        print(f"Сохранено: {pdb_path}")

if __name__ == "__main__":
    main()