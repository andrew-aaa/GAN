from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import *
from utils import set_seed, to_one_hot, write_metrics_row
from data.dataset import ToxinAntitoxinDataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.ema import EMA
from training.losses import gradient_penalty, token_ce_loss
from training.metrics import (
    aa_frequency_kl,
    ngram_diversity,
    eos_exact_rate,
    length_mae,
    nonempty_ratio,
    repeat_ratio,
)


def cfg(name: str, default):
    """Безопасное чтение новых параметров из config.py."""
    return globals().get(name, default)


def force_math_attention_for_wgan_gp() -> None:
    """
    WGAN-GP использует create_graph=True, то есть требует второй градиент через
    дискриминатор. На CUDA PyTorch может выбрать Flash/memory-efficient SDPA
    внутри TransformerEncoderLayer, а эти ядра не поддерживают нужный second-order
    backward. Поэтому для обучения с WGAN-GP включаем math SDPA.
    """
    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        print("[attention] Flash/mem-efficient SDPA disabled; math SDPA enabled for WGAN-GP.")
    except Exception as exc:
        print(f"[attention] Could not force math SDPA: {exc}")


def get_adv_weight(epoch_idx: int) -> float:
    if epoch_idx < GENERATOR_PRETRAIN_EPOCHS:
        return 0.0
    ramp_epochs = max(1, EPOCHS - GENERATOR_PRETRAIN_EPOCHS)
    progress = min(1.0, (epoch_idx - GENERATOR_PRETRAIN_EPOCHS + 1) / ramp_epochs)
    return ADV_WEIGHT_MAX * progress


def get_tau(epoch_idx: int) -> float:
    progress = min(1.0, epoch_idx / max(1, EPOCHS - 1))
    return TAU_START + (TAU_END - TAU_START) * progress


def length_control_losses(logits: torch.Tensor, target_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Явные differentiable-losses для управления концом последовательности.

    target устроен как [AA_1, ..., AA_N, EOS, PAD, ...].
    Поэтому EOS должен находиться в позиции target_lengths, аминокислоты — до неё,
    PAD — после неё.
    """
    batch_size, seq_len, _ = logits.shape
    device = logits.device
    target_lengths = target_lengths.clamp(min=0, max=seq_len - 1)

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    eos_mask = positions == target_lengths.unsqueeze(1)
    before_eos_mask = positions < target_lengths.unsqueeze(1)
    after_eos_mask = positions > target_lengths.unsqueeze(1)

    log_probs = F.log_softmax(logits, dim=-1)

    zero = logits.new_tensor(0.0)

    if eos_mask.any():
        eos_loss = -log_probs[..., EOS_IDX][eos_mask].mean()
    else:
        eos_loss = zero

    if after_eos_mask.any():
        pad_after_loss = -log_probs[..., PAD_IDX][after_eos_mask].mean()
    else:
        pad_after_loss = zero

    if before_eos_mask.any():
        # До EOS модель должна распределять массу только по 20 каноническим AA,
        # а не по PAD/BOS/EOS. Используем logsumexp как -log P(any amino acid).
        aa_log_mass = torch.logsumexp(log_probs[..., AA_START_IDX:], dim=-1)
        aa_before_loss = -aa_log_mass[before_eos_mask].mean()
    else:
        aa_before_loss = zero

    # Дополнительная мягкая защита: не генерировать BOS в target-позициях.
    bos_prob = log_probs[..., BOS_IDX].exp()
    not_pad_mask = positions <= target_lengths.unsqueeze(1)
    if not_pad_mask.any():
        bos_forbidden_loss = -torch.log1p(-bos_prob[not_pad_mask].clamp(max=1.0 - 1e-7)).mean()
    else:
        bos_forbidden_loss = zero

    return {
        "eos_loss": eos_loss,
        "pad_after_loss": pad_after_loss,
        "aa_before_loss": aa_before_loss,
        "bos_forbidden_loss": bos_forbidden_loss,
    }


def collect_metrics(fake_onehot: torch.Tensor, real_target: torch.Tensor, target_lengths: torch.Tensor) -> Dict[str, float]:
    fake_ids = fake_onehot.argmax(dim=-1).detach().cpu()
    real_ids = real_target.detach().cpu()
    target_lengths = target_lengths.detach().cpu()
    return {
        "nonempty": nonempty_ratio(fake_ids),
        "repeat_ratio": repeat_ratio(fake_ids),
        "ngram2": ngram_diversity(fake_ids, n=2),
        "ngram3": ngram_diversity(fake_ids, n=3),
        "aa_kl": aa_frequency_kl(fake_ids, real_ids),
        "len_mae": length_mae(fake_ids, target_lengths),
        "eos_exact": eos_exact_rate(fake_ids, target_lengths),
    }


def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {k: float(sum(x[k] for x in items) / len(items)) for k in keys}


@torch.no_grad()
def evaluate(generator: Generator, loader: DataLoader, epoch: int) -> dict[str, float]:
    generator.eval()
    tau = get_tau(epoch)
    metrics = []
    ce_vals = []
    len_vals = []
    eos_vals = []
    pad_vals = []
    aa_before_vals = []
    bos_vals = []

    for toxin_emb, decoder_input, target, aa_lengths in loader:
        toxin_emb = toxin_emb.to(DEVICE).float()
        decoder_input = decoder_input.to(DEVICE).long()
        target = target.to(DEVICE).long()
        aa_lengths = aa_lengths.to(DEVICE).long()

        z = torch.randn(toxin_emb.size(0), LATENT_DIM, device=DEVICE)
        logits, _ = generator.forward_teacher(decoder_input, toxin_emb, z=z, target_lengths=aa_lengths)
        fake_onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        length_terms = length_control_losses(logits, aa_lengths)

        ce_vals.append(float(token_ce_loss(logits, target).item()))
        len_vals.append(float(F.cross_entropy(generator.get_length_logits(toxin_emb), aa_lengths).item()))
        eos_vals.append(float(length_terms["eos_loss"].item()))
        pad_vals.append(float(length_terms["pad_after_loss"].item()))
        aa_before_vals.append(float(length_terms["aa_before_loss"].item()))
        bos_vals.append(float(length_terms["bos_forbidden_loss"].item()))
        metrics.append(collect_metrics(fake_onehot, target, aa_lengths))

    out = mean_dict(metrics)
    out["token_ce"] = float(sum(ce_vals) / max(1, len(ce_vals)))
    out["length_loss"] = float(sum(len_vals) / max(1, len(len_vals)))
    out["eos_loss"] = float(sum(eos_vals) / max(1, len(eos_vals)))
    out["pad_after_loss"] = float(sum(pad_vals) / max(1, len(pad_vals)))
    out["aa_before_loss"] = float(sum(aa_before_vals) / max(1, len(aa_before_vals)))
    out["bos_forbidden_loss"] = float(sum(bos_vals) / max(1, len(bos_vals)))

    out["proxy"] = (
        out["token_ce"]
        + cfg("PROXY_LENGTH_WEIGHT", 0.50) * out["length_loss"]
        + cfg("PROXY_EOS_WEIGHT", 0.50) * out["eos_loss"]
        + cfg("PROXY_PAD_WEIGHT", 0.20) * out["pad_after_loss"]
        + cfg("PROXY_AA_KL_WEIGHT", 0.20) * out["aa_kl"]
    )
    generator.train()
    return out


def main():
    set_seed(SEED)
    force_math_attention_for_wgan_gp()

    if cfg("RESET_METRICS_CSV", True):
        metrics_path = Path(METRICS_CSV_PATH)
        if metrics_path.exists():
            metrics_path.unlink()

    print(f"Using device: {DEVICE}")
    print(f"Loading toxin embeddings from: {TOXIN_EMBEDDINGS_PATH}")

    dataset = ToxinAntitoxinDataset(TOXIN_FASTA_PATH, ANTITOXIN_FASTA_PATH, TOXIN_EMBEDDINGS_PATH)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        raise ValueError("Слишком маленький датасет для train/val split.")

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    print(f"[split] train={len(train_ds)} val={len(val_ds)}")

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    ema = EMA(generator, decay=EMA_DECAY)

    optimizer_G = optim.AdamW(generator.parameters(), lr=LR_G, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=LR_D, betas=BETAS, weight_decay=WEIGHT_DECAY)

    best_val_proxy = math.inf
    fieldnames = [
        "epoch", "phase", "tau", "adv_w",
        "train_d_loss", "train_g_loss", "train_proxy",
        "train_token_ce", "train_length_loss", "train_eos_loss", "train_pad_after_loss", "train_aa_before_loss",
        "val_proxy", "val_token_ce", "val_length_loss", "val_eos_loss", "val_pad_after_loss", "val_aa_before_loss",
        "val_nonempty", "val_len_mae", "val_eos_exact", "val_ngram2", "val_ngram3", "val_aa_kl", "val_repeat_ratio",
        "ema_val_proxy",
    ]

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        adv_weight = get_adv_weight(epoch)
        tau = get_tau(epoch)

        batch_metrics = []
        d_losses, g_losses, proxies = [], [], []
        token_losses, length_losses, eos_losses, pad_losses, aa_before_losses = [], [], [], [], []

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch_idx, (toxin_emb, decoder_input, target, aa_lengths) in enumerate(progress):
            toxin_emb = toxin_emb.to(DEVICE).float()
            decoder_input = decoder_input.to(DEVICE).long()
            target = target.to(DEVICE).long()
            aa_lengths = aa_lengths.to(DEVICE).long()
            real_onehot = to_one_hot(target, VOCAB_SIZE).to(DEVICE)
            batch_size = toxin_emb.size(0)

            # ===== Train Discriminator =====
            if adv_weight > 0.0:
                # Для D fake строится без графа G: экономит память и не копит лишний autograd-граф.
                with torch.no_grad():
                    z_d = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                    logits_d, _ = generator.forward_teacher(decoder_input, toxin_emb, z=z_d, target_lengths=aa_lengths)
                    fake_onehot_d = F.gumbel_softmax(logits_d, tau=tau, hard=True, dim=-1)

                real_score = discriminator(toxin_emb, real_onehot, aa_lengths)
                fake_score = discriminator(toxin_emb, fake_onehot_d.detach(), aa_lengths)
                perm = torch.randperm(batch_size, device=DEVICE)
                mismatch_score = discriminator(toxin_emb[perm], real_onehot, aa_lengths)
                gp = gradient_penalty(discriminator, toxin_emb, real_onehot, fake_onehot_d.detach(), aa_lengths, DEVICE)

                fake_mix = (1.0 - MISMATCH_WEIGHT) * fake_score.mean() + MISMATCH_WEIGHT * mismatch_score.mean()
                loss_D = -(real_score.mean() - fake_mix) + LAMBDA_GP * gp

                optimizer_D.zero_grad(set_to_none=True)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
                optimizer_D.step()
                d_losses.append(float(loss_D.item()))

                del fake_onehot_d, logits_d, real_score, fake_score, mismatch_score, gp
            else:
                loss_D = torch.tensor(0.0, device=DEVICE)

            # ===== Train Generator =====
            should_update_g = (batch_idx % N_CRITIC == 0) or adv_weight == 0.0
            if should_update_g:
                z_g = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                logits_g, _ = generator.forward_teacher(decoder_input, toxin_emb, z=z_g, target_lengths=aa_lengths)
                fake_onehot_g = F.gumbel_softmax(logits_g, tau=tau, hard=True, dim=-1)

                token_loss = token_ce_loss(logits_g, target)
                length_loss = F.cross_entropy(generator.get_length_logits(toxin_emb), aa_lengths)
                length_terms = length_control_losses(logits_g, aa_lengths)

                adv_loss = torch.tensor(0.0, device=DEVICE)
                if adv_weight > 0.0:
                    adv_loss = -discriminator(toxin_emb, fake_onehot_g, aa_lengths).mean()

                loss_G = (
                    TOKEN_CE_WEIGHT * token_loss
                    + LENGTH_LOSS_WEIGHT * length_loss
                    + cfg("EOS_LOSS_WEIGHT", 0.75) * length_terms["eos_loss"]
                    + cfg("PAD_AFTER_EOS_WEIGHT", 0.35) * length_terms["pad_after_loss"]
                    + cfg("AA_BEFORE_EOS_WEIGHT", 0.25) * length_terms["aa_before_loss"]
                    + cfg("BOS_FORBIDDEN_WEIGHT", 0.05) * length_terms["bos_forbidden_loss"]
                    + adv_weight * adv_loss
                )

                optimizer_G.zero_grad(set_to_none=True)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_NORM)
                optimizer_G.step()
                ema.update(generator)

                cur_metrics = collect_metrics(fake_onehot_g, target, aa_lengths)
                proxy = float(
                    token_loss.item()
                    + cfg("PROXY_LENGTH_WEIGHT", 0.50) * length_loss.item()
                    + cfg("PROXY_EOS_WEIGHT", 0.50) * length_terms["eos_loss"].item()
                    + cfg("PROXY_PAD_WEIGHT", 0.20) * length_terms["pad_after_loss"].item()
                    + cfg("PROXY_AA_KL_WEIGHT", 0.20) * cur_metrics["aa_kl"]
                )

                batch_metrics.append(cur_metrics)
                g_losses.append(float(loss_G.item()))
                proxies.append(proxy)
                token_losses.append(float(token_loss.item()))
                length_losses.append(float(length_loss.item()))
                eos_losses.append(float(length_terms["eos_loss"].item()))
                pad_losses.append(float(length_terms["pad_after_loss"].item()))
                aa_before_losses.append(float(length_terms["aa_before_loss"].item()))

                progress.set_postfix(
                    g=f"{loss_G.item():.3f}",
                    d=f"{loss_D.item():.3f}",
                    proxy=f"{proxy:.3f}",
                    eos=f"{length_terms['eos_loss'].item():.3f}",
                    tau=f"{tau:.2f}",
                )

        train_metrics = mean_dict(batch_metrics) if batch_metrics else {
            "nonempty": 0.0, "repeat_ratio": 0.0, "ngram2": 0.0,
            "ngram3": 0.0, "aa_kl": 0.0, "len_mae": 0.0, "eos_exact": 0.0,
        }
        train_d_loss = float(sum(d_losses) / max(1, len(d_losses)))
        train_g_loss = float(sum(g_losses) / max(1, len(g_losses)))
        train_proxy = float(sum(proxies) / max(1, len(proxies))) if proxies else math.inf
        train_token_ce = float(sum(token_losses) / max(1, len(token_losses)))
        train_length_loss = float(sum(length_losses) / max(1, len(length_losses)))
        train_eos_loss = float(sum(eos_losses) / max(1, len(eos_losses)))
        train_pad_after_loss = float(sum(pad_losses) / max(1, len(pad_losses)))
        train_aa_before_loss = float(sum(aa_before_losses) / max(1, len(aa_before_losses)))

        val_metrics = evaluate(generator, val_loader, epoch)
        ema_val_metrics = evaluate(ema.shadow, val_loader, epoch)

        torch.save(generator.state_dict(), GENERATOR_LAST_PATH)
        torch.save(ema.state_dict(), EMA_LAST_PATH)
        if val_metrics["proxy"] < best_val_proxy:
            best_val_proxy = val_metrics["proxy"]
            torch.save(generator.state_dict(), GENERATOR_BEST_PATH)
            torch.save(ema.state_dict(), EMA_BEST_PATH)

        row = {
            "epoch": epoch + 1,
            "phase": "pretrain" if adv_weight == 0.0 else "hybrid",
            "tau": round(tau, 4),
            "adv_w": round(adv_weight, 4),
            "train_d_loss": round(train_d_loss, 6),
            "train_g_loss": round(train_g_loss, 6),
            "train_proxy": round(train_proxy, 6),
            "train_token_ce": round(train_token_ce, 6),
            "train_length_loss": round(train_length_loss, 6),
            "train_eos_loss": round(train_eos_loss, 6),
            "train_pad_after_loss": round(train_pad_after_loss, 6),
            "train_aa_before_loss": round(train_aa_before_loss, 6),
            "val_proxy": round(val_metrics["proxy"], 6),
            "val_token_ce": round(val_metrics["token_ce"], 6),
            "val_length_loss": round(val_metrics["length_loss"], 6),
            "val_eos_loss": round(val_metrics["eos_loss"], 6),
            "val_pad_after_loss": round(val_metrics["pad_after_loss"], 6),
            "val_aa_before_loss": round(val_metrics["aa_before_loss"], 6),
            "val_nonempty": round(val_metrics["nonempty"], 6),
            "val_len_mae": round(val_metrics["len_mae"], 6),
            "val_eos_exact": round(val_metrics["eos_exact"], 6),
            "val_ngram2": round(val_metrics["ngram2"], 6),
            "val_ngram3": round(val_metrics["ngram3"], 6),
            "val_aa_kl": round(val_metrics["aa_kl"], 6),
            "val_repeat_ratio": round(val_metrics["repeat_ratio"], 6),
            "ema_val_proxy": round(ema_val_metrics["proxy"], 6),
        }
        write_metrics_row(METRICS_CSV_PATH, fieldnames, row)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | phase: {row['phase']} | tau: {tau:.3f} | adv_w: {adv_weight:.3f} | "
            f"train_D: {train_d_loss:.4f} | train_G: {train_g_loss:.4f} | train_proxy: {train_proxy:.4f} | "
            f"val_proxy: {val_metrics['proxy']:.4f} | val_nonempty: {val_metrics['nonempty']:.2f} | "
            f"val_len_mae: {val_metrics['len_mae']:.2f} | val_eos_exact: {val_metrics['eos_exact']:.2f} | "
            f"val_eos_loss: {val_metrics['eos_loss']:.3f} | val_pad: {val_metrics['pad_after_loss']:.3f} | "
            f"val_ng2: {val_metrics['ngram2']:.3f} | val_ng3: {val_metrics['ngram3']:.3f} | "
            f"val_aa_kl: {val_metrics['aa_kl']:.4f} | val_repeat: {val_metrics['repeat_ratio']:.3f} | "
            f"ema_val_proxy: {ema_val_metrics['proxy']:.4f} | best_val_proxy: {best_val_proxy:.4f}"
        )

        if torch.cuda.is_available() and cfg("EMPTY_CACHE_EACH_EPOCH", False):
            torch.cuda.empty_cache()

    print(f"Training finished. Best checkpoint: {GENERATOR_BEST_PATH}")
    print(f"EMA best checkpoint: {EMA_BEST_PATH}")
    print(f"Metrics CSV: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
