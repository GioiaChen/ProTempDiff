#!/usr/bin/env python3
"""
Self-contained port of model_v11_final_fast.ipynb with optional 2-GPU DDP.
Run: torchrun --nproc_per_node=2 model.py
Single GPU / no torchrun: python model.py
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, Dataset as BaseDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# DDP helpers (minimal additions for multi-GPU)
# ---------------------------------------------------------------------------


def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def _ddp_setup() -> Tuple[int, int, int, torch.device, bool]:
    """Returns rank, local_rank, world_size, device, use_ddp."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is not available")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        use_ddp = world_size > 1
        return rank, local_rank, world_size, device, use_ddp
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return 0, 0, 1, device, False


def _ddp_cleanup(use_ddp: bool) -> None:
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _load_state_dict_into_model(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load ckpt saved without 'module.' into plain or DDP-wrapped model."""
    if isinstance(model, DDP) and state and not any(k.startswith("module.") for k in state):
        state = {"module." + k: v for k, v in state.items()}
    model.load_state_dict(state)


def _dist_sum_pair(loss_sum: float, n_batches: int, device: torch.device, use_ddp: bool) -> Tuple[float, int]:
    if not use_ddp:
        return loss_sum, n_batches
    t = torch.tensor([loss_sum, float(n_batches)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(round(t[1].item()))


# ---------------------------------------------------------------------------
# Paths (edit here)
# ---------------------------------------------------------------------------

# FINAL_H5_DIR = os.path.join(os.path.expanduser("~"), "path", "to", "final_h5")
# SPLIT_JSON = os.path.join(os.path.expanduser("~"), "path", "to", "split.json")
# SAVE_ROOT = os.path.join(os.path.expanduser("~"), "mdcath_project_final")
# CHECKPOINT_DIR = os.path.join(SAVE_ROOT, "checkpoints_diffusion", "v11_final_fast")
PROJECT_ROOT = r"..."
PREPROCESS_DIR = os.path.join(PROJECT_ROOT, "preprocess")
SPLIT_JSON = os.path.join(PREPROCESS_DIR, "split_1000.json")   # train / val lists
# FINAL_H5_DIR = os.path.join(PREPROCESS_DIR, "data")  # *_final.h5 flat folder
FINAL_H5_DIR = r"/dev/shm/data"
SAVE_ROOT = PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "ckpts/v2_fast")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_CHECKPOINT = None

TEMPERATURES = (320.0, 348.0, 379.0, 413.0, 450.0)
TEMP_TO_IDX = {float(t): i for i, t in enumerate(TEMPERATURES)}
NUM_TEMPS = len(TEMPERATURES)

AA20 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}


def build_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """[dist, dx, dy, dz] -> [E, 4] (same as v11)."""
    src, dst = edge_index
    rel = pos[dst] - pos[src]
    dist = torch.norm(rel, dim=-1, keepdim=True) + 1e-8
    return torch.cat([dist, rel], dim=-1)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FinalH5Dataset(BaseDataset):
    """FAST: H5 tensors in RAM; each sample is (protein, ti, fi) — slice pos in get()."""

    def __init__(self, h5_dir: str, protein_ids: List[str]):
        super().__init__()
        self.h5_dir = h5_dir
        self.protein_ids = list(protein_ids)
        self.samples: List[Dict[str, Any]] = []
        self.protein_refs: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not os.path.isdir(self.h5_dir):
            raise FileNotFoundError(self.h5_dir)
        for pid in self.protein_ids:
            path = os.path.join(self.h5_dir, f"{pid}_final.h5")
            if not os.path.isfile(path):
                warnings.warn(f"Missing H5: {path}")
                continue
            with h5py.File(path, "r") as f:
                pos = np.asarray(f["pos"][:], dtype=np.float32)
                aa_feat = np.asarray(f["aa_feat"][:], dtype=np.float32)
                pos_enc = np.asarray(f["pos_enc"][:], dtype=np.float32)
                edge_index = np.asarray(f["edge_index"][:], dtype=np.int64)
                pos_ref = np.asarray(f["pos_ref"][:], dtype=np.float32)
                temps = np.asarray(f["temps"][:], dtype=np.float32)

            if pos_enc.ndim == 1:
                pos_enc = pos_enc.reshape(-1, 1)
            node_feat = np.concatenate([aa_feat, pos_enc], axis=-1)

            n_t, n_fr, n_res, _ = pos.shape
            if n_res != node_feat.shape[0]:
                raise RuntimeError(f"{pid}: n_res mismatch pos vs node_feat")
            pos_t = torch.from_numpy(pos)
            ei = torch.from_numpy(edge_index)
            pref = torch.from_numpy(pos_ref)
            nf = torch.from_numpy(node_feat)

            self.protein_refs[pid] = {
                "pos": pos_t,
                "node_feat": nf,
                "pos_ref": pref,
                "edge_index": ei,
                "num_nodes": n_res,
            }

            for ti in range(n_t):
                T = float(temps[ti])
                if T not in TEMP_TO_IDX:
                    continue
                t_idx = TEMP_TO_IDX[T]
                for fi in range(n_fr):
                    self.samples.append({
                        "protein_id": pid,
                        "temperature": T,
                        "temp_idx": t_idx,
                        "ti": ti,
                        "fi": fi,
                    })

        if not self.samples:
            raise RuntimeError(f"No samples under {self.h5_dir} for given ids")

        print(
            f"[FinalH5Dataset] {len(self.samples)} samples | {len(self.protein_refs)} proteins "
            f"(index-backed pos tensor per protein)"
        )

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        s = self.samples[idx]
        ref = self.protein_refs[s["protein_id"]]
        pos = ref["pos"][s["ti"], s["fi"]].clone()
        edge_attr = build_edge_attr(pos, ref["edge_index"])
        data = Data(
            x=ref["node_feat"],
            pos=pos,
            pos_ref=ref["pos_ref"],
            edge_index=ref["edge_index"],
            edge_attr=edge_attr,
        )
        data.temp_idx = torch.tensor([s["temp_idx"]], dtype=torch.long)
        data.T = torch.tensor([s["temperature"]], dtype=torch.float32)
        data.protein_id = s["protein_id"]
        return data


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    val_batch_size: Optional[int] = 8
    num_workers: int = 4
    use_amp: bool = True
    use_torch_compile: bool = False
    hidden_dim: int = 128
    num_layers: int = 6
    time_emb_dim: int = 64
    temp_emb_dim: int = 32
    num_timesteps: int = 1000
    noise_schedule: str = "cosine"
    num_epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    warmup_epochs: int = 5
    early_stop_patience: int = 8
    aux_dist_weight: float = 0.0
    log_every: int = 1
    log_every_n_batches: int = 500  # per-batch progress (e.g. batch 100/500); 0 = off
    coord_scale: float = 10.0
    ddim_steps: int = 50
    num_gen_samples: int = 40
    max_train_proteins: Optional[int] = None
    max_val_proteins: Optional[int] = None
    save_every_epochs: int = 5
    resume_checkpoint: Optional[str] = None


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EGNNLayer(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, edge_attr_dim: int = 4,
                 cond_dim: int = 0, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)
        self.has_cond = cond_dim > 0
        if self.has_cond:
            self.film_gamma = nn.Linear(cond_dim, node_dim)
            self.film_beta = nn.Linear(cond_dim, node_dim)
            nn.init.ones_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        self.coord_scale_logit = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, x, edge_index, edge_attr=None, cond=None):
        if edge_index.numel() == 0:
            return h, x
        src, dst = edge_index
        rel = x[dst] - x[src]
        dist_sq = (rel ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        dist = dist_sq.sqrt()
        edge_inputs = [h[dst], h[src], dist_sq]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
        msg = self.edge_mlp(torch.cat(edge_inputs, dim=-1))
        msg = msg.to(dtype=h.dtype)
        agg = h.new_zeros(h.size(0), msg.size(1))
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        h_in = torch.cat([h, agg], dim=-1)
        if self.has_cond and cond is not None:
            gamma = self.film_gamma(cond)
            beta = self.film_beta(cond)
            h_modulated = gamma * h + beta
            h_in = torch.cat([h_modulated, agg], dim=-1)
        h_out = self.node_mlp(h_in)
        if self.residual:
            h_out = h + h_out
        h_out = self.norm(h_out)
        w = self.coord_mlp(msg)
        weighted = (rel / (dist + 1.0) * w).to(dtype=x.dtype)
        delta = x.new_zeros(x.size(0), 3)
        delta.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)
        deg = x.new_zeros(x.size(0), 1)
        deg.scatter_add_(0, dst.unsqueeze(-1), x.new_ones(dst.size(0), 1))
        delta = delta / deg.clamp(min=1.0)
        scale = torch.sigmoid(self.coord_scale_logit)
        x_out = x + scale * delta
        return h_out, x_out


class EGNNStack(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, edge_attr_dim=4, cond_dim=0):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim=node_dim, hidden_dim=hidden_dim,
                      edge_attr_dim=edge_attr_dim, cond_dim=cond_dim, residual=True)
            for _ in range(num_layers)
        ])

    def forward(self, h, x, edge_index, edge_attr=None, cond=None):
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr=edge_attr, cond=cond)
        return h, x


class ConformationDenoiser(nn.Module):
    def __init__(self, node_input_dim, hidden_dim=256, num_layers=8,
                 num_temps=NUM_TEMPS, time_emb_dim=64, temp_emb_dim=32, edge_attr_dim=4):
        super().__init__()
        self.time_net = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temp_embed = nn.Embedding(num_temps, temp_emb_dim)
        self.temp_net = nn.Sequential(
            nn.Linear(temp_emb_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ref_dist_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.egnn = EGNNStack(node_dim=hidden_dim, hidden_dim=hidden_dim,
                              num_layers=num_layers, edge_attr_dim=edge_attr_dim,
                              cond_dim=hidden_dim)

    def forward(self, x_noisy, t, node_feat, edge_index, temp_idx, batch_idx,
                edge_attr=None, pos_ref=None):
        t_emb = self.time_net(t)
        temp_emb = self.temp_net(self.temp_embed(temp_idx.squeeze(-1)))
        cond = t_emb + temp_emb
        cond_per_node = cond[batch_idx]
        if pos_ref is not None:
            ref_dist = (x_noisy - pos_ref).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
            ref_emb = self.ref_dist_net(ref_dist)
            cond_per_node = cond_per_node + ref_emb
        h = self.node_proj(torch.cat([node_feat, cond_per_node], dim=-1))
        _, x_out = self.egnn(h, x_noisy, edge_index, edge_attr=edge_attr, cond=cond_per_node)
        return x_out - x_noisy

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, schedule="cosine", beta_start=1e-4, beta_end=0.02):
        self.T = num_timesteps
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(schedule)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.sched = {
            "betas": betas.float(), "alphas": alphas.float(),
            "alpha_bar": alpha_bar.float(),
            "sqrt_alpha_bar": alpha_bar.sqrt().float(),
            "sqrt_1m_alpha_bar": (1.0 - alpha_bar).sqrt().float(),
            "sqrt_recip_alpha": (1.0 / alphas).sqrt().float(),
            "posterior_var": (betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).float(),
        }

    @staticmethod
    def _cosine_beta_schedule(T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-5, 0.999)

    def _v(self, name, t, batch_idx):
        vals = self.sched[name].to(t.device)
        return vals[t.long()][batch_idx].unsqueeze(-1)

    @staticmethod
    def zero_com(x, batch_idx):
        n_graphs = int(batch_idx.max().item() + 1)
        sums = x.new_zeros(n_graphs, 3)
        sums.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(x), x)
        ones = x.new_ones(batch_idx.shape[0], 1)
        counts = x.new_zeros(n_graphs, 1)
        counts.scatter_add_(0, batch_idx.unsqueeze(-1), ones)
        centers = sums / counts.clamp(min=1.0)
        return x - centers[batch_idx]

    def q_sample(self, x0, t, batch_idx, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            noise = self.zero_com(noise, batch_idx)
        a = self._v("sqrt_alpha_bar", t, batch_idx)
        b = self._v("sqrt_1m_alpha_bar", t, batch_idx)
        return a * x0 + b * noise, noise

    def compute_loss(self, model, batch, aux_dist_weight=0.0, coord_scale=10.0):
        x0_raw = self.zero_com(batch.pos, batch.batch)
        x0 = x0_raw / coord_scale
        pos_ref_scaled = self.zero_com(batch.pos_ref, batch.batch) / coord_scale
        B = batch.num_graphs
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        noise = self.zero_com(noise, batch.batch)
        xt, _ = self.q_sample(x0, t, batch.batch, noise)
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        eps_pred = model(
            x_noisy=xt, t=t, node_feat=batch.x, edge_index=batch.edge_index,
            temp_idx=batch.temp_idx, batch_idx=batch.batch,
            edge_attr=edge_attr, pos_ref=pos_ref_scaled,
        )
        mse_loss = F.mse_loss(eps_pred, noise)
        aux_loss = torch.tensor(0.0, device=x0.device)
        if aux_dist_weight > 0:
            ab = self._v("sqrt_alpha_bar", t, batch.batch)
            sb = self._v("sqrt_1m_alpha_bar", t, batch.batch)
            x0_pred = (xt - sb * eps_pred) / ab.clamp(min=1e-5)
            for g in range(B):
                mask = batch.batch == g
                dp = torch.cdist(x0_pred[mask], x0_pred[mask])
                dt = torch.cdist(x0[mask], x0[mask])
                aux_loss = aux_loss + F.mse_loss(dp, dt)
            aux_loss = aux_loss / B
        total = mse_loss + aux_dist_weight * aux_loss
        return total, {"mse": float(mse_loss.item()), "aux_dist": float(aux_loss.item())}

    @torch.no_grad()
    def ddim_sample(self, model, node_feat, edge_index, temp_idx, batch_idx,
                    num_nodes, ddim_steps=50, eta=0.0,
                    edge_attr=None, pos_ref=None, coord_scale=10.0, verbose=False):
        model.eval()
        device = node_feat.device
        out_dtype = node_feat.dtype
        step_seq = torch.linspace(0, self.T - 1, ddim_steps + 1, dtype=torch.long, device=device).flip(0)
        x = torch.randn(num_nodes, 3, device=device, dtype=torch.float64)
        x = self.zero_com(x, batch_idx)
        alpha_bar = self.sched["alpha_bar"].to(device=device, dtype=torch.float64)
        pos_ref_scaled = pos_ref / coord_scale if pos_ref is not None else None
        for i in range(ddim_steps):
            t_cur, t_prev = step_seq[i], step_seq[i + 1]
            B = int(batch_idx.max().item() + 1)
            t_batch = torch.full((B,), t_cur.item(), device=device, dtype=torch.long)
            eps_pred = model(
                x.to(out_dtype), t_batch, node_feat, edge_index, temp_idx, batch_idx,
                edge_attr=edge_attr, pos_ref=pos_ref_scaled,
            ).to(torch.float64)
            ab_cur = alpha_bar[t_cur]
            ab_prev = alpha_bar[t_prev] if t_prev >= 0 else x.new_tensor(1.0)
            sqrt_ab = ab_cur.sqrt().clamp(min=1e-4)
            if ab_cur.item() > 1e-4:
                x0_pred = (x - (1.0 - ab_cur).sqrt() * eps_pred) / sqrt_ab
                x0_pred = x0_pred.clamp(-10.0, 10.0)
                dir_xt = (1.0 - ab_prev).clamp(min=0.0).sqrt() * eps_pred
                x = ab_prev.sqrt() * x0_pred + dir_xt
            else:
                x = x - (ab_cur.sqrt() - ab_prev.sqrt()) * eps_pred
            x = self.zero_com(x, batch_idx)
            if verbose and (i % max(ddim_steps // 5, 1) == 0):
                print(f"  DDIM {i}/{ddim_steps} t={t_cur.item()}")
        return (x * coord_scale).to(out_dtype)


class GaussianDiffusionFast(GaussianDiffusion):
    """AMP-friendly loss: model forward under autocast when use_amp."""

    def compute_loss(self, model, batch, aux_dist_weight=0.0, coord_scale=10.0, use_amp: bool = False):
        x0_raw = self.zero_com(batch.pos, batch.batch)
        x0 = x0_raw / coord_scale
        pos_ref_scaled = self.zero_com(batch.pos_ref, batch.batch) / coord_scale
        B = batch.num_graphs
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        noise = self.zero_com(noise, batch.batch)
        xt, _ = self.q_sample(x0, t, batch.batch, noise)
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        amp_on = bool(use_amp) and torch.cuda.is_available()
        with torch.autocast(device_type="cuda", enabled=amp_on):
            eps_pred = model(
                x_noisy=xt, t=t, node_feat=batch.x, edge_index=batch.edge_index,
                temp_idx=batch.temp_idx, batch_idx=batch.batch,
                edge_attr=edge_attr, pos_ref=pos_ref_scaled,
            )
        mse_loss = F.mse_loss(eps_pred.float(), noise.float())
        aux_loss = torch.tensor(0.0, device=x0.device)
        if aux_dist_weight > 0:
            ab = self._v("sqrt_alpha_bar", t, batch.batch)
            sb = self._v("sqrt_1m_alpha_bar", t, batch.batch)
            x0_pred = (xt - sb * eps_pred.float()) / ab.clamp(min=1e-5)
            for g in range(B):
                mask = batch.batch == g
                dp = torch.cdist(x0_pred[mask], x0_pred[mask])
                dt = torch.cdist(x0[mask], x0[mask])
                aux_loss = aux_loss + F.mse_loss(dp, dt)
            aux_loss = aux_loss / B
        total = mse_loss + aux_dist_weight * aux_loss
        return total, {"mse": float(mse_loss.item()), "aux_dist": float(aux_loss.item())}


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}
        self._backup = {}

    def step(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        self._backup = {n: p.clone() for n, p in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self._backup[name])
        self._backup = {}


class TrainerFast:
    """Trainer + AMP; DDP: sync metrics, rank0 save, unwrapped state_dict."""

    def __init__(
        self,
        model,
        diffusion,
        train_loader,
        val_loader,
        cfg,
        device,
        save_dir,
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.save_dir = save_dir
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.is_master = rank == 0

        os.makedirs(save_dir, exist_ok=True)
        self.use_amp = bool(getattr(cfg, "use_amp", False)) and torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.warmup_epochs > 0:
            warmup = LambdaLR(self.optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / cfg.warmup_epochs))
            cosine = CosineAnnealingLR(self.optimizer, T_max=max(cfg.num_epochs - cfg.warmup_epochs, 1), eta_min=1e-6)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(cfg.num_epochs, 1), eta_min=1e-6)
        self.ema = EMA(_unwrap_model(self.model), decay=cfg.ema_decay)
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self._resume_start_epoch = 1
        self._resume_wait = 0
        rp = getattr(cfg, "resume_checkpoint", None)
        if rp and os.path.isfile(rp):
            self._load_training_state(rp)
        elif rp and self.is_master:
            print(f"[Resume] file not found, training from scratch: {rp}")

    def _load_training_state(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _load_state_dict_into_model(self.model, ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_shadow" in ckpt:
            for k, v in ckpt["ema_shadow"].items():
                nk = k[7:] if k.startswith("module.") else k
                if nk in self.ema.shadow:
                    self.ema.shadow[nk] = v.to(self.device).clone()
        if self.scaler is not None and "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.history = ckpt.get("history", {"train_loss": [], "val_loss": [], "lr": []})
        done = int(ckpt.get("epoch", 0))
        self._resume_start_epoch = done + 1
        self._resume_wait = int(ckpt.get("wait", 0))
        if self.is_master:
            print(
                f"[Resume] {path} | last completed epoch={done} | "
                f"next epoch={self._resume_start_epoch} | early-stop wait={self._resume_wait}"
            )

    def _run_epoch(self, loader, train, epoch: int, phase: str, num_epochs: int):
        self.model.train(train)
        total_loss, n = 0.0, 0
        use_loss_amp = self.use_amp
        n_batches = len(loader)
        log_b = int(getattr(self.cfg, "log_every_n_batches", 0) or 0)
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self.diffusion.compute_loss(
                self.model, batch,
                aux_dist_weight=self.cfg.aux_dist_weight if train else 0.0,
                coord_scale=self.cfg.coord_scale,
                use_amp=use_loss_amp,
            )
            if train:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                self.ema.step(_unwrap_model(self.model))
            total_loss += float(loss.item())
            n += 1
            if (
                self.is_master
                and log_b > 0
                and n_batches > 0
                and ((i + 1) % log_b == 0 or (i + 1) == n_batches)
            ):
                print(
                    f"[Epoch {epoch}/{num_epochs} | {phase}] batch {i + 1}/{n_batches}",
                    flush=True,
                )
        return total_loss, n

    def save_checkpoint(self, name, epoch, wait=0):
        if not self.is_master:
            return
        payload = {
            "epoch": epoch,
            "model_state": _unwrap_model(self.model).state_dict(),
            "ema_shadow": {k: v.detach().cpu() for k, v in self.ema.shadow.items()},
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "wait": wait,
            "history": self.history,
            "config": vars(self.cfg),
            "node_input_dim": NODE_INPUT_DIM,
        }
        if self.scaler is not None:
            payload["scaler_state"] = self.scaler.state_dict()
        torch.save(payload, os.path.join(self.save_dir, name))

    def fit(self, num_epochs):
        patience = self.cfg.early_stop_patience
        wait = self._resume_wait
        start_epoch = self._resume_start_epoch
        if start_epoch > num_epochs:
            if self.is_master:
                print(f"[Resume] start_epoch {start_epoch} > num_epochs {num_epochs}; nothing to run.")
            return self.history
        for epoch in range(start_epoch, num_epochs + 1):
            if self.use_ddp and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            tr_sum, tr_n = self._run_epoch(self.train_loader, True, epoch, "train", num_epochs)
            tr_sum, tr_n = _dist_sum_pair(tr_sum, tr_n, self.device, self.use_ddp)
            train_loss = tr_sum / max(tr_n, 1)
            self.history["train_loss"].append(train_loss)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.ema.apply(_unwrap_model(self.model))
            va_sum, va_n = self._run_epoch(self.val_loader, False, epoch, "val", num_epochs)
            va_sum, va_n = _dist_sum_pair(va_sum, va_n, self.device, self.use_ddp)
            val_loss = va_sum / max(va_n, 1)
            self.ema.restore(_unwrap_model(self.model))
            self.history["val_loss"].append(val_loss)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt", epoch, wait=wait)
                wait = 0
            else:
                wait += 1

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(lr)
            se = getattr(self.cfg, "save_every_epochs", 0) or 0
            if se > 0 and epoch % se == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, wait=wait)

            if self.is_master and epoch % self.cfg.log_every == 0:
                tag = " *" if improved else ""
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | train {train_loss:.5f} | val {val_loss:.5f} | "
                    f"lr {lr:.2e} | {time.time()-t0:.1f}s | patience {wait}/{patience}{tag}"
                )

            if patience > 0 and wait >= patience:
                if self.is_master:
                    print(f"[Early stop] epoch {epoch}")
                break

        self.save_checkpoint("final.pt", epoch, wait=wait)
        if self.is_master:
            print(f"Best val: {self.best_val_loss:.6f}")
        return self.history


NODE_INPUT_DIM: int = 0


def main() -> None:
    global NODE_INPUT_DIM

    rank, local_rank, world_size, device, use_ddp = _ddp_setup()
    is_master = rank == 0

    if is_master:
        print(f"Python : {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
        print(f"DDP: {use_ddp} | world_size={world_size} | rank={rank} | device={device}")

    tcfg = TrainConfig(resume_checkpoint=RESUME_CHECKPOINT)

    with open(SPLIT_JSON, "r", encoding="utf-8") as fp:
        split = json.load(fp)
    train_ids = list(split.get("train", []))
    val_ids = list(split.get("val", []))
    if not train_ids or not val_ids:
        raise ValueError("split.json needs non-empty 'train' and 'val'")
    if tcfg.max_train_proteins is not None:
        train_ids = train_ids[: tcfg.max_train_proteins]
    if tcfg.max_val_proteins is not None:
        val_ids = val_ids[: tcfg.max_val_proteins]
    if is_master:
        print(f"Split train={len(train_ids)} val={len(val_ids)}")

    set_seed(tcfg.seed + rank)
    train_ds = FinalH5Dataset(FINAL_H5_DIR, train_ids)
    val_ds = FinalH5Dataset(FINAL_H5_DIR, val_ids)

    NODE_INPUT_DIM = int(train_ds[0].x.shape[1])
    if is_master:
        print("NODE_INPUT_DIM:", NODE_INPUT_DIM)

    _dl_kw = {
        "num_workers": tcfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if tcfg.num_workers > 0:
        _dl_kw["persistent_workers"] = True

    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds, batch_size=tcfg.batch_size, shuffle=False, sampler=train_sampler, **_dl_kw
        )
        _val_bs = tcfg.val_batch_size if tcfg.val_batch_size is not None else tcfg.batch_size
        val_loader = DataLoader(
            val_ds, batch_size=_val_bs, shuffle=False, sampler=val_sampler, **_dl_kw
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=tcfg.batch_size, shuffle=True, **_dl_kw
        )
        _val_bs = tcfg.val_batch_size if tcfg.val_batch_size is not None else tcfg.batch_size
        val_loader = DataLoader(val_ds, batch_size=_val_bs, shuffle=False, **_dl_kw)

    if is_master:
        print(
            f"Batches train/ rank0={len(train_loader)} val/ rank0={len(val_loader)} | "
            f"batch_size={tcfg.batch_size} (per GPU) | num_workers={tcfg.num_workers} | "
            f"global batch ≈ {tcfg.batch_size * world_size} (train)"
        )

    model = ConformationDenoiser(
        node_input_dim=NODE_INPUT_DIM,
        hidden_dim=tcfg.hidden_dim,
        num_layers=tcfg.num_layers,
        num_temps=NUM_TEMPS,
        time_emb_dim=tcfg.time_emb_dim,
        temp_emb_dim=tcfg.temp_emb_dim,
        edge_attr_dim=4,
    ).to(device)

    if getattr(tcfg, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if is_master:
                print("[Fast] torch.compile(model) enabled")
        except Exception as e:
            if is_master:
                print("[Fast] torch.compile skipped:", e)

    raw_model = model
    if use_ddp:
        # find_unused_parameters=True: EGNN may skip layers (e.g. empty edge_index path); avoids DDP reducer errors.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    diffusion = GaussianDiffusionFast(num_timesteps=tcfg.num_timesteps, schedule=tcfg.noise_schedule)
    if is_master:
        print(f"Params: {_unwrap_model(model).param_count():,} | T={diffusion.T} | AMP={tcfg.use_amp}")

    # Sanity on unwrapped module only: no_grad forward on DDP without backward breaks reducer state.
    raw_model.eval()
    batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        loss, info = diffusion.compute_loss(
            raw_model, batch, coord_scale=tcfg.coord_scale, use_amp=tcfg.use_amp and torch.cuda.is_available()
        )
    if is_master:
        print(f"Sanity loss: {loss.item():.4f} | {info}")
    model.train()

    trainer = TrainerFast(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=tcfg,
        device=device,
        save_dir=CHECKPOINT_DIR,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )
    history = trainer.fit(tcfg.num_epochs)

    if is_master:
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(epochs, history["train_loss"], label="train")
        ax.plot(epochs, history["val_loss"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved loss curve: {plot_path}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _load_state_dict_into_model(model, ckpt["model_state"])
        model.eval()
        print("Loaded", ckpt_path, "epoch", ckpt.get("epoch"))

        raw = _unwrap_model(model)
        raw.eval()

        DDIM_STEPS = 50
        template = val_ds[0]
        protein_id = template.protein_id
        temperature = float(template.T.item())
        print(f"Template protein={protein_id} T={temperature}K")

        template_batch = Batch.from_data_list([template]).to(device)
        pos_ref_raw = diffusion.zero_com(template_batch.pos_ref, template_batch.batch).to(device)
        trainer.ema.apply(raw)

        gen_frames = []
        t0 = time.time()
        for i in range(tcfg.num_gen_samples):
            gen_pos = diffusion.ddim_sample(
                model=raw,
                node_feat=template_batch.x,
                edge_index=template_batch.edge_index,
                temp_idx=template_batch.temp_idx,
                batch_idx=template_batch.batch,
                num_nodes=template_batch.x.size(0),
                ddim_steps=DDIM_STEPS,
                eta=0.0,
                edge_attr=template_batch.edge_attr,
                pos_ref=pos_ref_raw,
                coord_scale=tcfg.coord_scale,
                verbose=False,
            )
            gen_frames.append(gen_pos.cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"  generated {i+1}/{tcfg.num_gen_samples}")
        print(f"DDIM-{DDIM_STEPS} done in {time.time()-t0:.1f}s")
        trainer.ema.restore(raw)

        md_frames = []
        for s in val_ds.samples:
            if s["protein_id"] == protein_id and s["temperature"] == temperature:
                ref = val_ds.protein_refs[s["protein_id"]]
                p = ref["pos"][s["ti"], s["fi"]].numpy()
                md_frames.append(p - p.mean(0))
        print(f"MD frames for same (pid,T): {len(md_frames)} | generated: {len(gen_frames)}")

        if md_frames and gen_frames:
            ref = np.mean(np.stack(md_frames, axis=0), axis=0)
            g0 = gen_frames[0] - gen_frames[0].mean(0)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=8, alpha=0.5, label="mean MD")
            ax.scatter(g0[:, 0], g0[:, 1], g0[:, 2], s=8, alpha=0.5, label="gen0")
            ax.legend()
            ax.set_title(f"{protein_id} @ {temperature}K")
            plt.tight_layout()
            overlay_path = os.path.join(CHECKPOINT_DIR, "overlay_3d.png")
            plt.savefig(overlay_path, dpi=150)
            plt.close()
            print(f"Saved 3D overlay: {overlay_path}")
        else:
            print("Skip plot: missing md or gen frames")

    if use_ddp:
        dist.barrier()

    _ddp_cleanup(use_ddp)


if __name__ == "__main__":
    main()
