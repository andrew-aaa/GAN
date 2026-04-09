from __future__ import annotations

import math
from copy import deepcopy
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
from training.metrics import aa_frequency_kl, ngram_diversity, eos_exact_rate, length_mae, nonempty_ratio, repeat_ratio


def get_adv_weight(epoch_idx: int) -> float:
    if epoch_idx < GENERATOR_PRETRAIN_EPOCHS:
        return 0.0
    ramp_epochs = max(1, EPOCHS - GENERATOR_PRETRAIN_EPOCHS)
    progress = min(1.0, (epoch_idx - GENERATOR_PRETRAIN_EPOCHS + 1) / ramp_epochs)
    return ADV_WEIGHT_MAX * progress


def get_tau(epoch_idx: int) -> float:
    progress = min(1.0, epoch_idx / max(1, EPOCHS - 1))
    return TAU_START + (TAU_END - TAU_START) * progress


def collect_metrics(fake_onehot: torch.Tensor, real_target: torch.Tensor, target_lengths: torch.Tensor) -> Dict[str, float]:
    fake_ids = fake_onehot.argmax(dim=-1).detach().cpu()
    real_ids = real_target.detach().cpu()
    target_lengths = target_lengths.detach().cpu()
    return {
        'nonempty': nonempty_ratio(fake_ids),
        'repeat_ratio': repeat_ratio(fake_ids),
        'ngram2': ngram_diversity(fake_ids, n=2),
        'ngram3': ngram_diversity(fake_ids, n=3),
        'aa_kl': aa_frequency_kl(fake_ids, real_ids),
        'len_mae': length_mae(fake_ids, target_lengths),
        'eos_exact': eos_exact_rate(fake_ids, target_lengths),
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
    for toxin_emb, decoder_input, target, aa_lengths in loader:
        toxin_emb = toxin_emb.to(DEVICE).float()
        decoder_input = decoder_input.to(DEVICE).long()
        target = target.to(DEVICE).long()
        aa_lengths = aa_lengths.to(DEVICE).long()
        z = torch.randn(toxin_emb.size(0), LATENT_DIM, device=DEVICE)
        logits, _ = generator.forward_teacher(decoder_input, toxin_emb, z=z, target_lengths=aa_lengths)
        fake_onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        ce_vals.append(float(token_ce_loss(logits, target).item()))
        len_vals.append(float(F.cross_entropy(generator.get_length_logits(toxin_emb), aa_lengths).item()))
        metrics.append(collect_metrics(fake_onehot, target, aa_lengths))
    out = mean_dict(metrics)
    out['token_ce'] = float(sum(ce_vals) / max(1, len(ce_vals)))
    out['length_loss'] = float(sum(len_vals) / max(1, len(len_vals)))
    out['proxy'] = out['token_ce'] + 0.5 * out['length_loss'] + 0.2 * out['aa_kl']
    generator.train()
    return out


def main():
    set_seed(SEED)
    print(f'Using device: {DEVICE}')
    print(f'Loading toxin embeddings from: {TOXIN_EMBEDDINGS_PATH}')

    dataset = ToxinAntitoxinDataset(TOXIN_FASTA_PATH, ANTITOXIN_FASTA_PATH, TOXIN_EMBEDDINGS_PATH)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    print(f'[split] train={len(train_ds)} val={len(val_ds)}')

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
        'epoch', 'phase', 'tau', 'adv_w', 'train_d_loss', 'train_g_loss', 'train_proxy',
        'val_proxy', 'val_token_ce', 'val_length_loss', 'val_nonempty', 'val_len_mae',
        'val_eos_exact', 'val_ngram2', 'val_ngram3', 'val_aa_kl', 'val_repeat_ratio'
    ]

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        adv_weight = get_adv_weight(epoch)
        tau = get_tau(epoch)

        batch_metrics = []
        d_losses, g_losses, proxies = [], [], []

        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
        for batch_idx, (toxin_emb, decoder_input, target, aa_lengths) in enumerate(progress):
            toxin_emb = toxin_emb.to(DEVICE).float()
            decoder_input = decoder_input.to(DEVICE).long()
            target = target.to(DEVICE).long()
            aa_lengths = aa_lengths.to(DEVICE).long()
            real_onehot = to_one_hot(target, VOCAB_SIZE).to(DEVICE)
            batch_size = toxin_emb.size(0)
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)

            # Fast fake pass: one transformer pass via teacher forcing
            logits, _ = generator.forward_teacher(decoder_input, toxin_emb, z=z, target_lengths=aa_lengths)
            fake_onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

            if adv_weight > 0.0:
                real_score = discriminator(toxin_emb, real_onehot, aa_lengths)
                fake_score = discriminator(toxin_emb, fake_onehot.detach(), aa_lengths)
                perm = torch.randperm(batch_size, device=DEVICE)
                mismatch_score = discriminator(toxin_emb[perm], real_onehot, aa_lengths)
                gp = gradient_penalty(discriminator, toxin_emb, real_onehot, fake_onehot.detach(), aa_lengths, DEVICE)
                fake_mix = (1.0 - MISMATCH_WEIGHT) * fake_score.mean() + MISMATCH_WEIGHT * mismatch_score.mean()
                loss_D = -(real_score.mean() - fake_mix) + LAMBDA_GP * gp
                optimizer_D.zero_grad(set_to_none=True)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
                optimizer_D.step()
                d_losses.append(float(loss_D.item()))
            else:
                loss_D = torch.tensor(0.0, device=DEVICE)

            should_update_g = (batch_idx % N_CRITIC == 0) or adv_weight == 0.0
            if should_update_g:
                token_loss = token_ce_loss(logits, target)
                length_loss = F.cross_entropy(generator.get_length_logits(toxin_emb), aa_lengths)
                adv_loss = torch.tensor(0.0, device=DEVICE)
                if adv_weight > 0.0:
                    adv_loss = -discriminator(toxin_emb, fake_onehot, aa_lengths).mean()
                loss_G = TOKEN_CE_WEIGHT * token_loss + LENGTH_LOSS_WEIGHT * length_loss + adv_weight * adv_loss
                optimizer_G.zero_grad(set_to_none=True)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_NORM)
                optimizer_G.step()
                ema.update(generator)

                cur_metrics = collect_metrics(fake_onehot, target, aa_lengths)
                proxy = float(token_loss.item() + 0.5 * length_loss.item() + 0.2 * cur_metrics['aa_kl'])
                batch_metrics.append(cur_metrics)
                g_losses.append(float(loss_G.item()))
                proxies.append(proxy)
                progress.set_postfix(g=f'{loss_G.item():.3f}', d=f'{loss_D.item():.3f}', proxy=f'{proxy:.3f}', tau=f'{tau:.2f}')

        train_metrics = mean_dict(batch_metrics) if batch_metrics else {
            'nonempty': 0.0, 'repeat_ratio': 0.0, 'ngram2': 0.0, 'ngram3': 0.0, 'aa_kl': 0.0, 'len_mae': 0.0, 'eos_exact': 0.0
        }
        train_d_loss = float(sum(d_losses) / max(1, len(d_losses)))
        train_g_loss = float(sum(g_losses) / max(1, len(g_losses)))
        train_proxy = float(sum(proxies) / max(1, len(proxies))) if proxies else math.inf

        val_metrics = evaluate(generator, val_loader, epoch)
        ema_val_metrics = evaluate(ema.shadow, val_loader, epoch)

        torch.save(generator.state_dict(), GENERATOR_LAST_PATH)
        torch.save(ema.state_dict(), EMA_LAST_PATH)
        if val_metrics['proxy'] < best_val_proxy:
            best_val_proxy = val_metrics['proxy']
            torch.save(generator.state_dict(), GENERATOR_BEST_PATH)
            torch.save(ema.state_dict(), EMA_BEST_PATH)

        row = {
            'epoch': epoch + 1,
            'phase': 'pretrain' if adv_weight == 0.0 else 'hybrid',
            'tau': round(tau, 4),
            'adv_w': round(adv_weight, 4),
            'train_d_loss': round(train_d_loss, 6),
            'train_g_loss': round(train_g_loss, 6),
            'train_proxy': round(train_proxy, 6),
            'val_proxy': round(val_metrics['proxy'], 6),
            'val_token_ce': round(val_metrics['token_ce'], 6),
            'val_length_loss': round(val_metrics['length_loss'], 6),
            'val_nonempty': round(val_metrics['nonempty'], 6),
            'val_len_mae': round(val_metrics['len_mae'], 6),
            'val_eos_exact': round(val_metrics['eos_exact'], 6),
            'val_ngram2': round(val_metrics['ngram2'], 6),
            'val_ngram3': round(val_metrics['ngram3'], 6),
            'val_aa_kl': round(val_metrics['aa_kl'], 6),
            'val_repeat_ratio': round(val_metrics['repeat_ratio'], 6),
        }
        write_metrics_row(METRICS_CSV_PATH, fieldnames, row)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | phase: {row['phase']} | tau: {tau:.3f} | adv_w: {adv_weight:.3f} | "
            f"train_D: {train_d_loss:.4f} | train_G: {train_g_loss:.4f} | train_proxy: {train_proxy:.4f} | "
            f"val_proxy: {val_metrics['proxy']:.4f} | val_nonempty: {val_metrics['nonempty']:.2f} | "
            f"val_len_mae: {val_metrics['len_mae']:.2f} | val_eos_exact: {val_metrics['eos_exact']:.2f} | "
            f"val_ng2: {val_metrics['ngram2']:.3f} | val_ng3: {val_metrics['ngram3']:.3f} | "
            f"val_aa_kl: {val_metrics['aa_kl']:.4f} | val_repeat: {val_metrics['repeat_ratio']:.3f} | "
            f"ema_val_proxy: {ema_val_metrics['proxy']:.4f} | best_val_proxy: {best_val_proxy:.4f}"
        )

    print(f'Training finished. Best checkpoint: {GENERATOR_BEST_PATH}')
    print(f'EMA best checkpoint: {EMA_BEST_PATH}')
    print(f'Metrics CSV: {METRICS_CSV_PATH}')


if __name__ == '__main__':
    main()
