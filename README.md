# Antidote GAN — Colab-friendly version

Это версия проекта, переделанная так, чтобы обучение не "замирало" на первой эпохе в Google Colab.

## Что изменено

- Убрана самая дорогая часть: autoregressive `sample()` больше **не используется внутри train loop**.
- Во время обучения fake-последовательности строятся через **один teacher-forcing проход** генератора.
- Добавлены:
  - `tqdm`-прогресс по батчам;
  - `train/val split`;
  - усреднение метрик по эпохе;
  - логирование в `logs/metrics.csv`;
  - `mismatched pairs` для усиления conditional GAN;
  - `projection discriminator`;
  - корректная маска по длине, а не через `argmax(one-hot)`;
  - `temperature annealing` в обучении.

## Почему раньше казалось, что код завис

В старой схеме `generator.sample()` вызывался прямо в обучении. Он генерировал последовательность **пошагово**, и на каждом шаге заново прогонял трансформер по всей длине. На CPU или даже в слабом Colab-GPU это делало первую эпоху очень долгой.

## Как запускать в Colab

```python
!pip install fair-esm biopython tqdm
```

Скопируй проект в `/content/GAN`, затем:

```python
%cd /content/GAN
!python prepare_pairs.py
!python precompute_toxin_embeddings.py
!python train.py
```

## Проверка GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
```

Если `False`, значит Colab сейчас реально на CPU.

## Инференс

```python
!python inference/generate_antidote.py
```
