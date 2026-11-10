"""Transformer model, lazy sequence dataset, and one train/eval epoch runner."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class StockSeqDataset(Dataset):
    """Lazily slice windows: store each stock's full series, cut windows on access."""

    def __init__(self, arrays_by_code, samples, seq_len):
        self.arrays = arrays_by_code
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, end = self.samples[idx]
        feat, ret, _dates = self.arrays[code]
        window = np.ascontiguousarray(feat[end - self.seq_len + 1: end + 1])
        y = ret[end]
        return torch.from_numpy(window), torch.tensor(y, dtype=torch.float32)


class StockTransformer(nn.Module):
    def __init__(self, n_features, seq_len, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, L, F)
        x = self.input_proj(x) + self.pos_enc[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])   # take the last time step
        return self.head(x).squeeze(-1)


def run_epoch(model, loader, opt, loss_fn, *, device, use_amp, scaler, target_scale,
              train=True, desc=""):
    """Run one epoch.

    During eval we keep *all* predictions (even rows with NaN target) so they
    stay aligned with sample order for cross-sectional ranking.
    """
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0
    preds: List[np.ndarray] = []
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = ((yb - 1.0) * target_scale).to(device, non_blocking=True)   # regress excess return
        mask = ~torch.isnan(yb)
        if train:
            if not mask.any():
                continue
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(xb)
                loss = loss_fn(out[mask], yb[mask])
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * int(mask.sum())
            total_n += int(mask.sum())
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model(xb)
                if mask.any():
                    loss = loss_fn(out[mask], yb[mask])
                    total_loss += loss.item() * int(mask.sum())
                    total_n += int(mask.sum())
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            preds.append(out.float().detach().cpu().numpy() / target_scale)
    avg = total_loss / max(total_n, 1)
    if train:
        return avg, None
    return avg, (np.concatenate(preds) if preds else np.array([]))


def build_model(n_features: int, cfg) -> StockTransformer:
    return StockTransformer(
        n_features, cfg.seq_len, cfg.d_model, cfg.nhead,
        cfg.num_layers, cfg.dim_ff, cfg.dropout,
    )


def make_loader(dataset, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.resolved_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )
