import sys
from types import ModuleType
import numpy as np

# ==== FIX numpy ====
if not hasattr(np, "BUFSIZE"):
    np.BUFSIZE = 8192

# ==== FIX torch._six ====
try:
    import torch._six
except ImportError:
    torch_six = ModuleType("torch._six")
    torch_six.inf = float("inf")
    sys.modules["torch._six"] = torch_six


# ==== 🔥 КЛЮЧЕВОЙ ФИКС esmfold ====

# создаем фейковый модуль ДО импорта esm
fake_trunk = ModuleType("esm.esmfold.v1.trunk")

class StructureModuleConfig:
    def __init__(self, *args, **kwargs):
        pass

class IPAConfig:
    def __init__(self, *args, **kwargs):
        pass

fake_trunk.StructureModuleConfig = StructureModuleConfig
fake_trunk.IPAConfig = IPAConfig

# подсовываем в систему
sys.modules["esm.esmfold.v1.trunk"] = fake_trunk


# ==== теперь можно импортировать esm ====
import torch
import esm
import pandas as pd
from pathlib import Path
import os

# 5. Остальной код (функция main и т.д.)

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
    
    model.set_chunk_size(64)

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