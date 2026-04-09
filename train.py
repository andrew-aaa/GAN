from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    GENERATOR_PRETRAIN_EPOCHS,
    N_CRITIC,
    VAL_SPLIT,
    SEED,
    LR_G,
    LR_D,
    BETAS,
    WEIGHT_DECAY,
    LATENT_DIM,
    VOCAB_SIZE,
    MODEL_SAVE_PATH,
    BEST_PROXY_MODEL_SAVE_PATH,
    EMA_MODEL_SAVE_PATH,
    BEST_EMA_PROXY_MODEL_SAVE_PATH,
    METRICS_CSV_PATH,
    TOXIN_FASTA_PATH,
    ANTITOXIN_FASTA_PATH,
    TOXIN_EMBEDDINGS_PATH,
    LAMBDA_GP,
    MISMATCH_WEIGHT,
    EMA_DECAY,
    GRAD_CLIP_NORM,
    ADV_WEIGHT_MAX,
    TOKEN_CE_WEIGHT,
    LENGTH_LOSS_WEIGHT,
    TAU_START,
    TAU_END,
)
from utils import to_one_hot
from data.dataset import ToxinAntitoxinDataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.ema import EMA
from training.losses import gradient_penalty, token_ce_loss
from training.metrics import (
    aa_frequency_kl,
    ngram_diversity,
    eos_exact_rate,
    valid_eos_pad_rate,
    length_mae,
    nonempty_ratio,
    repeat_ratio,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ema_forward(ema: EMA, *args, **kwargs):
    return ema.shadow(*args, **kwargs)


def get_adv_weight(epoch_idx: int) -> float:
    if epoch_idx < GENERATOR_PRETRAIN_EPOCHS:
        return 0.0
    ramp_epochs = max(1, EPOCHS - GENERATOR_PRETRAIN_EPOCHS)
    progress = min(1.0, (epoch_idx - GENERATOR_PRETRAIN_EPOCHS + 1) / ramp_epochs)
    return ADV_WEIGHT_MAX * progress


def get_sampling_temperature(epoch_idx: int) -> float:
    if EPOCHS <= 1:
        return TAU_END
    progress = epoch_idx / (EPOCHS - 1)
    return TAU_START + (TAU_END - TAU_START) * progress


def metrics_template() -> dict[str, float]:
    return {
        "nonempty": 0.0,
        "valid": 0.0,
        "repeat_ratio": 0.0,
        "ngram2": 0.0,
        "ngram3": 0.0,
        "aa_kl": 0.0,
        "len_mae": 0.0,
        "eos_exact": 0.0,
    }


def evaluate_batch_metrics(fake_onehot: torch.Tensor, real_target: torch.Tensor, target_lengths: torch.Tensor):
    fake_ids = fake_onehot.argmax(dim=-1)
    real_ids = real_target
    return {
        "nonempty": nonempty_ratio(fake_ids),
        "valid": valid_eos_pad_rate(fake_ids),
        "repeat_ratio": repeat_ratio(fake_ids),
        "ngram2": ngram_diversity(fake_ids, n=2),
        "ngram3": ngram_diversity(fake_ids, n=3),
        "aa_kl": aa_frequency_kl(fake_ids, real_ids),
        "len_mae": length_mae(fake_ids, target_lengths),
        "eos_exact": eos_exact_rate(fake_ids, target_lengths),
    }


def aggregate_metrics(acc: dict[str, float], metrics: dict[str, float]):
    for key, value in metrics.items():
        acc[key] += float(value)


def average_metrics(acc: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        return {k: 0.0 for k in acc}
    return {k: v / count for k, v in acc.items()}


@torch.no_grad()
def evaluate_model(generator_like, loader: DataLoader, sampling_temperature: float, adv_weight: float, use_ema: bool = False):
    metrics_acc = metrics_template()
    proxy_sum = 0.0
    batch_count = 0

    for toxin_emb, decoder_input, target, aa_lengths in loader:
        toxin_emb = toxin_emb.to(DEVICE).float()
        decoder_input = decoder_input.to(DEVICE).long()
        target = target.to(DEVICE).long()
        aa_lengths = aa_lengths.to(DEVICE).long()

        z = torch.randn(toxin_emb.size(0), LATENT_DIM, device=DEVICE)

        if use_ema:
            logits, _ = generator_like.forward_teacher(decoder_input, toxin_emb, z=z, target_lengths=aa_lengths)
            fake_onehot, _ = generator_like.sample(
                toxin_emb,
                z=z,
                target_lengths=aa_lengths,
                sampling_temperature=sampling_temperature,
                gumbel_tau=1.0,
                hard=True,
            )
            length_logits = generator_like.get_length_logits(toxin_emb)
        else:
            logits, _ = generator_like.forward_teacher(decoder_input, toxin_emb, z=z, target_lengths=aa_lengths)
            fake_onehot, _ = generator_like.sample(
                toxin_emb,
                z=z,
                target_lengths=aa_lengths,
                sampling_temperature=sampling_temperature,
                gumbel_tau=1.0,
                hard=True,
            )
            length_logits = generator_like.get_length_logits(toxin_emb)

        token_loss = token_ce_loss(logits, target)
        length_loss = F.cross_entropy(length_logits, aa_lengths)
        batch_metrics = evaluate_batch_metrics(fake_onehot, target, aa_lengths)

        proxy = float(token_loss.item() + 0.5 * length_loss.item() + 0.2 * batch_metrics["aa_kl"])
        proxy_sum += proxy
        aggregate_metrics(metrics_acc, batch_metrics)
        batch_count += 1

    return proxy_sum / max(1, batch_count), average_metrics(metrics_acc, batch_count)


def write_metrics_csv(rows: list[dict[str, float | int | str]]):
    path = Path(METRICS_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Loading toxin embeddings from: {TOXIN_EMBEDDINGS_PATH}")

    dataset = ToxinAntitoxinDataset(
        TOXIN_FASTA_PATH,
        ANTITOXIN_FASTA_PATH,
        TOXIN_EMBEDDINGS_PATH,
    )

    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("После разбиения не осталось обучающих примеров. Уменьши VAL_SPLIT.")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    ema = EMA(generator, decay=EMA_DECAY)

    optimizer_G = optim.AdamW(generator.parameters(), lr=LR_G, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=LR_D, betas=BETAS, weight_decay=WEIGHT_DECAY)

    best_val_proxy = float("inf")
    csv_rows: list[dict[str, float | int | str]] = []

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()

        adv_weight = get_adv_weight(epoch)
        sampling_temperature = get_sampling_temperature(epoch)

        train_loss_d_sum = 0.0
        train_loss_g_sum = 0.0
        train_metrics_acc = metrics_template()
        d_updates = 0
        g_updates = 0

        for batch_idx, (toxin_emb, decoder_input, target, aa_lengths) in enumerate(train_loader):
            toxin_emb = toxin_emb.to(DEVICE).float()
            decoder_input = decoder_input.to(DEVICE).long()
            target = target.to(DEVICE).long()
            aa_lengths = aa_lengths.to(DEVICE).long()
            real_onehot = to_one_hot(target, VOCAB_SIZE).to(DEVICE)
            batch_size = toxin_emb.size(0)

            # ===== Critic =====
            if adv_weight > 0.0:
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_onehot, _ = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=aa_lengths,
                    sampling_temperature=sampling_temperature,
                    gumbel_tau=1.0,
                    hard=True,
                )

                real_score = discriminator(toxin_emb, real_onehot, aa_lengths)
                fake_score = discriminator(toxin_emb, fake_onehot.detach(), aa_lengths)

                if batch_size > 1:
                    perm = torch.randperm(batch_size, device=DEVICE)
                    mismatch_toxin_emb = toxin_emb[perm]
                    mismatch_score = discriminator(mismatch_toxin_emb, real_onehot, aa_lengths)
                    fake_mix = 0.5 * fake_score.mean() + MISMATCH_WEIGHT * mismatch_score.mean()
                else:
                    fake_mix = fake_score.mean()

                gp = gradient_penalty(
                    discriminator,
                    toxin_emb,
                    real_onehot,
                    fake_onehot.detach(),
                    aa_lengths,
                    DEVICE,
                )
                loss_D = -(real_score.mean() - fake_mix) + LAMBDA_GP * gp

                optimizer_D.zero_grad(set_to_none=True)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
                optimizer_D.step()

                train_loss_d_sum += float(loss_D.item())
                d_updates += 1

            # ===== Generator =====
            should_update_g = (batch_idx % N_CRITIC == 0) or adv_weight == 0.0
            if should_update_g:
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                logits, _ = generator.forward_teacher(
                    decoder_input,
                    toxin_emb,
                    z=z,
                    target_lengths=aa_lengths,
                )
                token_loss = token_ce_loss(logits, target)
                length_logits = generator.get_length_logits(toxin_emb)
                length_loss = F.cross_entropy(length_logits, aa_lengths)

                adv_loss = torch.tensor(0.0, device=DEVICE)
                fake_onehot, _ = generator.sample(
                    toxin_emb,
                    z=z,
                    target_lengths=aa_lengths,
                    sampling_temperature=sampling_temperature,
                    gumbel_tau=1.0,
                    hard=True,
                )
                if adv_weight > 0.0:
                    fake_score = discriminator(toxin_emb, fake_onehot, aa_lengths)
                    adv_loss = -fake_score.mean()

                loss_G = (
                    TOKEN_CE_WEIGHT * token_loss
                    + LENGTH_LOSS_WEIGHT * length_loss
                    + adv_weight * adv_loss
                )

                optimizer_G.zero_grad(set_to_none=True)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_NORM)
                optimizer_G.step()
                ema.update(generator)

                batch_metrics = evaluate_batch_metrics(fake_onehot.detach(), target, aa_lengths)
                aggregate_metrics(train_metrics_acc, batch_metrics)
                train_loss_g_sum += float(loss_G.item())
                g_updates += 1

        generator.eval()
        ema.shadow.eval()

        train_metrics = average_metrics(train_metrics_acc, g_updates)
        train_loss_d = train_loss_d_sum / max(1, d_updates)
        train_loss_g = train_loss_g_sum / max(1, g_updates)

        val_proxy, val_metrics = evaluate_model(generator, val_loader, sampling_temperature, adv_weight, use_ema=False)
        ema_val_proxy, ema_val_metrics = evaluate_model(ema.shadow, val_loader, sampling_temperature, adv_weight, use_ema=True)

        torch.save(generator.state_dict(), MODEL_SAVE_PATH)
        torch.save(ema.state_dict(), EMA_MODEL_SAVE_PATH)

        if ema_val_proxy < best_val_proxy:
            best_val_proxy = ema_val_proxy
            torch.save(generator.state_dict(), BEST_PROXY_MODEL_SAVE_PATH)
            torch.save(ema.state_dict(), BEST_EMA_PROXY_MODEL_SAVE_PATH)

        row = {
            "epoch": epoch + 1,
            "phase": "pretrain" if adv_weight == 0.0 else "hybrid",
            "adv_w": round(adv_weight, 6),
            "sample_temp": round(sampling_temperature, 6),
            "train_d_loss": round(train_loss_d, 6),
            "train_g_loss": round(train_loss_g, 6),
            "train_nonempty": round(train_metrics["nonempty"], 6),
            "train_valid": round(train_metrics["valid"], 6),
            "train_len_mae": round(train_metrics["len_mae"], 6),
            "train_eos_exact": round(train_metrics["eos_exact"], 6),
            "train_ng2": round(train_metrics["ngram2"], 6),
            "train_ng3": round(train_metrics["ngram3"], 6),
            "train_aa_kl": round(train_metrics["aa_kl"], 6),
            "train_repeat": round(train_metrics["repeat_ratio"], 6),
            "val_proxy": round(val_proxy, 6),
            "val_nonempty": round(val_metrics["nonempty"], 6),
            "val_valid": round(val_metrics["valid"], 6),
            "val_len_mae": round(val_metrics["len_mae"], 6),
            "val_eos_exact": round(val_metrics["eos_exact"], 6),
            "val_ng2": round(val_metrics["ngram2"], 6),
            "val_ng3": round(val_metrics["ngram3"], 6),
            "val_aa_kl": round(val_metrics["aa_kl"], 6),
            "val_repeat": round(val_metrics["repeat_ratio"], 6),
            "ema_val_proxy": round(ema_val_proxy, 6),
            "ema_val_nonempty": round(ema_val_metrics["nonempty"], 6),
            "ema_val_valid": round(ema_val_metrics["valid"], 6),
            "ema_val_len_mae": round(ema_val_metrics["len_mae"], 6),
            "ema_val_eos_exact": round(ema_val_metrics["eos_exact"], 6),
            "ema_val_ng2": round(ema_val_metrics["ngram2"], 6),
            "ema_val_ng3": round(ema_val_metrics["ngram3"], 6),
            "ema_val_aa_kl": round(ema_val_metrics["aa_kl"], 6),
            "ema_val_repeat": round(ema_val_metrics["repeat_ratio"], 6),
            "best_ema_val_proxy": round(best_val_proxy, 6),
        }
        csv_rows.append(row)
        write_metrics_csv(csv_rows)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"phase: {row['phase']} | "
            f"adv_w: {adv_weight:.3f} | "
            f"temp: {sampling_temperature:.3f} | "
            f"D_loss: {train_loss_d:.4f} | "
            f"G_loss: {train_loss_g:.4f} | "
            f"val_proxy: {val_proxy:.4f} | "
            f"ema_val_proxy: {ema_val_proxy:.4f} | "
            f"best_ema_val_proxy: {best_val_proxy:.4f} | "
            f"val_valid: {val_metrics['valid']:.2f} | "
            f"val_nonempty: {val_metrics['nonempty']:.2f} | "
            f"val_len_mae: {val_metrics['len_mae']:.2f} | "
            f"val_ng2: {val_metrics['ngram2']:.3f} | "
            f"val_ng3: {val_metrics['ngram3']:.3f} | "
            f"val_aa_kl: {val_metrics['aa_kl']:.4f}"
        )

    print(f"Training finished. Latest generator saved to: {MODEL_SAVE_PATH}")
    print(f"Latest EMA generator saved to: {EMA_MODEL_SAVE_PATH}")
    print(f"Best EMA proxy checkpoint saved to: {BEST_EMA_PROXY_MODEL_SAVE_PATH}")
    print(f"Metrics CSV saved to: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
