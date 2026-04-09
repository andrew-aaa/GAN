# Improved antidote GAN

В проект добавлены приоритетные улучшения из аудита:
- **projection-discriminator** вместо простой конкатенации признаков;
- **mismatched-pairs** в обучении критика, чтобы дискриминатор реально учитывал условие токсина;
- **mask по длине**, а не через `argmax(one-hot)`, что корректнее для WGAN-GP;
- **dropout убран из дискриминатора** для более стабильного сигнала критика;
- **validation split** и усреднение метрик по эпохе;
- логирование в **`logs/metrics.csv`**;
- **temperature annealing** в обучении;
- корректный **temperature-sampling на инференсе** через масштабирование логитов;
- строгий разбор ID в `prepare_pairs.py` через регулярные выражения.

## Порядок запуска

1. Положить исходные FASTA в `data/`:
   - `type_II_T_exp.fas`
   - `type_II_AT_exp.fas`
2. Подготовить пары:
   ```bash
   python prepare_pairs.py
   ```
3. Предвычислить эмбеддинги токсинов:
   ```bash
   python precompute_toxin_embeddings.py
   ```
4. Обучить модель:
   ```bash
   python train.py
   ```
5. Сгенерировать кандидата:
   ```bash
   python generate_antidote.py
   ```

## Что теперь логируется

В `logs/metrics.csv` сохраняются:
- train/val/ema_val proxy;
- train/val валидность EOS/PAD;
- length MAE;
- n-gram diversity (`ng2`, `ng3`);
- KL по частотам аминокислот;
- repeat ratio;
- текущий вес adversarial-компонента и температура семплирования.

## Структура
- `data/dataset.py` — датасет с предвычисленными ESM-эмбеддингами;
- `models/generator.py` — autoregressive generator с teacher forcing;
- `models/discriminator.py` — projection-discriminator с conditioning по токсину;
- `training/losses.py` — WGAN-GP и supervised loss;
- `training/metrics.py` — валидность / diversity / KL / length метрики;
- `training/ema.py` — EMA для генератора;
- `inference/generate_antidote.py` — инференс и выбор лучшего кандидата.
