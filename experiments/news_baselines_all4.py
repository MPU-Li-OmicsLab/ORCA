# -*- coding: utf-8 -*-
"""
news_baselines_all4.py

Baselines (DRNet / VCNet-style / TARNet) for News covariates under 4 tasks:
  1) contT_contY   (MISE, RMSE_grid)
  2) contT_binY    (MISE over prob curve, RMSE_grid)
  3) binT_contY    (sqrt-PEHE, ATE error)
  4) binT_binY     (sqrt-PEHE, ATE error)

Semi-synthetic DGP on top of News covariates X with conf_strength in {0.1, 1.0, 5.0}.
- No external proprietary code.
- GitHub-friendly: argparse, deterministic seeds, CSV logging, resume.

Usage examples:
  python news_baselines_all4.py --data_path /path/news.npz --out_dir ./runs/news_baselines
  FAST=1 python news_baselines_all4.py --data_path /path/news.npz --out_dir ./runs/news_baselines_fast
  python news_baselines_all4.py --data_path /path/news.csv --out_dir ./runs/news_baselines --label_col y

Expected data formats:
  (A) NPZ preferred:
      - x_train (n_tr,d), x_test (n_te,d)    OR
      - x (n,d)                             OR
      - x (n,d,rep)  (optional rep dimension)
  (B) CSV:
      - all columns as features by default; use --label_col to drop a label column.

Outputs (per task):
  TASK_<task>/raw_<task>.csv
  TASK_<task>/summary_<task>.csv

Notes:
  - For contT tasks, we evaluate dose-response on a fixed T_GRID.
  - For binT tasks, we evaluate ITE on test set via mu1-mu0 (or p1-p0 for binY).

"""

import os, time, math, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ==========================================================
# Utils
# ==========================================================
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit_np(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)

def rmse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def sqrt_pehe(ite_hat, ite_true):
    ite_hat = np.asarray(ite_hat, dtype=np.float32)
    ite_true = np.asarray(ite_true, dtype=np.float32)
    return float(np.sqrt(np.mean((ite_hat - ite_true) ** 2)))

def mise_grid(mu_hat, mu_true, t_grid):
    # mu_hat/mu_true: (n,G)
    dt = float(t_grid[1] - t_grid[0])
    mu_hat = np.asarray(mu_hat, dtype=np.float32)
    mu_true = np.asarray(mu_true, dtype=np.float32)
    return float(np.mean(np.sum((mu_hat - mu_true) ** 2, axis=1) * dt))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_done_keys(raw_csv):
    if not os.path.exists(raw_csv):
        return set(), []
    df = pd.read_csv(raw_csv, engine="python", on_bad_lines="skip")
    keys = set(zip(df.task, df.conf_strength, df.rep_id, df.seed, df.method))
    wall = df["wall_time_sec"].to_numpy(dtype=float) if "wall_time_sec" in df.columns else np.array([], dtype=float)
    return keys, wall.tolist()

def append_row(raw_csv, row: dict):
    df1 = pd.DataFrame([row])
    if not os.path.exists(raw_csv):
        df1.to_csv(raw_csv, index=False)
    else:
        df1.to_csv(raw_csv, mode="a", header=False, index=False)


# ==========================================================
# Data loader (News X only)
# ==========================================================
def _standardize_train_test(Xtr, Xte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    return Xtr_s, Xte_s

def load_news_X(data_path: str, rep_id: int, seed: int, test_ratio: float, label_col: str | None):
    """
    Return standardized (Xtr, Xte).
    - If NPZ has x_train/x_test: use them
    - Else if NPZ has x: use random split (rep_id/seed drives split)
    - If x is (n,d,rep): pick rep_id
    - If CSV: read dataframe, optionally drop label_col, then split
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".npz":
        z = np.load(data_path)
        keys = set(z.keys())

        if ("x_train" in keys) and ("x_test" in keys):
            Xtr = z["x_train"].astype(np.float32)
            Xte = z["x_test"].astype(np.float32)
            return _standardize_train_test(Xtr, Xte)

        if "x" not in keys:
            raise ValueError(f"NPZ missing x / x_train+x_test. keys={list(keys)}")

        X = z["x"].astype(np.float32)
        if X.ndim == 3:
            # (n,d,rep)
            rep_max = X.shape[2]
            rid = int(rep_id) % rep_max
            X = X[:, :, rid]

        # random split
        rs = int(seed + 1337 + rep_id * 17)
        Xtr, Xte = train_test_split(X, test_size=test_ratio, random_state=rs, shuffle=True)
        return _standardize_train_test(Xtr, Xte)

    # CSV fallback
    df = pd.read_csv(data_path)
    if label_col is not None and label_col in df.columns:
        df = df.drop(columns=[label_col])
    X = df.to_numpy(dtype=np.float32)

    rs = int(seed + 1337 + rep_id * 17)
    Xtr, Xte = train_test_split(X, test_size=test_ratio, random_state=rs, shuffle=True)
    return _standardize_train_test(Xtr, Xte)


# ==========================================================
# Semi-synthetic DGP for 4 tasks on top of X
#   conf_strength controls confounding intensity in T assignment
# ==========================================================
def make_score_from_x(X, seed, k=8):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    k = min(k, d)
    w = rng.normal(0, 1, size=(k,)).astype(np.float32)
    return (X[:, :k] @ w).astype(np.float32)

def make_propensity_from_x(X, conf_strength, seed):
    score = make_score_from_x(X, seed=seed, k=8)
    e = sigmoid_np(conf_strength * score).astype(np.float32)
    return np.clip(e, 1e-6, 1 - 1e-6)

def g_of_t(t):
    # smooth map R -> (0,1)
    return sigmoid_np(1.25 * t).astype(np.float32)

def dgp_build(task, Xtr, Xte, conf_strength, seed, t_min, t_max, t_grid):
    """
    Returns dict with fields needed by each task.
    """
    rng = np.random.default_rng(seed + int(conf_strength * 1000) + 202)
    d = Xtr.shape[1]

    # shared components (heterogeneity)
    # baseline b(x) and two "endpoints" mu0(x), mu1(x) (continuous latent)
    x0_tr = Xtr[:, 0]
    x1_tr = Xtr[:, 1] if d > 1 else Xtr[:, 0]
    x0_te = Xte[:, 0]
    x1_te = Xte[:, 1] if d > 1 else Xte[:, 0]

    # latent endpoints (continuous)
    mu0_tr = (1.0 + 0.5 * x0_tr + 0.25 * (x1_tr ** 2)).astype(np.float32)
    mu1_tr = (mu0_tr + 1.0 + 0.75 * np.sin(2.0 * x0_tr) + 0.25 * x1_tr).astype(np.float32)

    mu0_te = (1.0 + 0.5 * x0_te + 0.25 * (x1_te ** 2)).astype(np.float32)
    mu1_te = (mu0_te + 1.0 + 0.75 * np.sin(2.0 * x0_te) + 0.25 * x1_te).astype(np.float32)

    def mu_cont(mu0, mu1, t):
        g = g_of_t(t)
        return (mu0 + g * (mu1 - mu0)).astype(np.float32)

    if task.startswith("binT_"):
        # binary treatment
        e_tr = make_propensity_from_x(Xtr, conf_strength, seed=seed + 11)
        e_te = make_propensity_from_x(Xte, conf_strength, seed=seed + 19)

        Ttr = (rng.uniform(0, 1, size=Xtr.shape[0]) < e_tr).astype(np.float32)
        Tte = (rng.uniform(0, 1, size=Xte.shape[0]) < e_te).astype(np.float32)

        if task.endswith("_contY"):
            Ytr = np.where(Ttr < 0.5, mu0_tr, mu1_tr).astype(np.float32) + rng.normal(0, 1.0, size=Xtr.shape[0]).astype(np.float32)
            ite_true = (mu1_te - mu0_te).astype(np.float32)
            ate_true = float(np.mean(ite_true))
            return dict(
                Xtr=Xtr, Ttr=Ttr, Ytr=Ytr,
                Xte=Xte, Tte=Tte,
                mu0_te=mu0_te.astype(np.float32), mu1_te=mu1_te.astype(np.float32),
                ite_true=ite_true, ate_true=ate_true
            )

        # binT_binY: convert endpoints to probs
        # stabilize then sigmoid
        m = float(np.mean(np.concatenate([mu0_tr, mu1_tr], axis=0)))
        s = float(np.std(np.concatenate([mu0_tr, mu1_tr], axis=0)) + 1e-6)
        p0_tr = sigmoid_np((mu0_tr - m) / s).astype(np.float32)
        p1_tr = sigmoid_np((mu1_tr - m) / s).astype(np.float32)
        p0_te = sigmoid_np((mu0_te - m) / s).astype(np.float32)
        p1_te = sigmoid_np((mu1_te - m) / s).astype(np.float32)

        p0_tr = np.clip(p0_tr, 1e-6, 1 - 1e-6)
        p1_tr = np.clip(p1_tr, 1e-6, 1 - 1e-6)
        p0_te = np.clip(p0_te, 1e-6, 1 - 1e-6)
        p1_te = np.clip(p1_te, 1e-6, 1 - 1e-6)

        Ytr = np.empty((Xtr.shape[0],), dtype=np.float32)
        u = rng.uniform(0, 1, size=Xtr.shape[0]).astype(np.float32)
        Ytr[Ttr < 0.5] = (u[Ttr < 0.5] < p0_tr[Ttr < 0.5]).astype(np.float32)
        Ytr[Ttr >= 0.5] = (u[Ttr >= 0.5] < p1_tr[Ttr >= 0.5]).astype(np.float32)

        ite_true = (p1_te - p0_te).astype(np.float32)
        ate_true = float(np.mean(ite_true))
        return dict(
            Xtr=Xtr, Ttr=Ttr, Ytr=Ytr,
            Xte=Xte, Tte=Tte,
            mu0_te=p0_te.astype(np.float32), mu1_te=p1_te.astype(np.float32),
            ite_true=ite_true, ate_true=ate_true
        )

    # continuous treatment tasks
    score_tr = make_score_from_x(Xtr, seed=seed + 31, k=8)
    score_te = make_score_from_x(Xte, seed=seed + 37, k=8)

    Ttr = (conf_strength * score_tr + rng.normal(0, 1.0, size=Xtr.shape[0]).astype(np.float32)).astype(np.float32)
    Tte = (conf_strength * score_te + rng.normal(0, 1.0, size=Xte.shape[0]).astype(np.float32)).astype(np.float32)

    # normalize + clip
    Ttr = (Ttr - Ttr.mean()) / (Ttr.std() + 1e-6) * 1.5
    Tte = (Tte - Tte.mean()) / (Tte.std() + 1e-6) * 1.5
    Ttr = np.clip(Ttr, t_min, t_max).astype(np.float32)
    Tte = np.clip(Tte, t_min, t_max).astype(np.float32)

    if task == "contT_contY":
        mu_tr = mu_cont(mu0_tr, mu1_tr, Ttr)
        Ytr = mu_tr + rng.normal(0, 1.0, size=Xtr.shape[0]).astype(np.float32)

        mu_grid_te = np.stack([mu_cont(mu0_te, mu1_te, np.full_like(mu0_te, t, dtype=np.float32)) for t in t_grid], axis=1)
        return dict(Xtr=Xtr, Ttr=Ttr, Ytr=Ytr, Xte=Xte, Tte=Tte, mu_grid_te=mu_grid_te.astype(np.float32))

    # contT_binY
    # use mu as logit (stabilized) -> prob -> Bernoulli
    mu_tr = mu_cont(mu0_tr, mu1_tr, Ttr)
    mu_center = float(mu_tr.mean())
    mu_scale = float(mu_tr.std() + 1e-6)
    logit_tr = ((mu_tr - mu_center) / mu_scale * 1.0).astype(np.float32)
    p_tr = sigmoid_np(logit_tr).astype(np.float32)
    Ytr = (rng.uniform(0, 1, size=Xtr.shape[0]) < p_tr).astype(np.float32)

    p_grid_te = []
    for t in t_grid:
        mu_t = mu_cont(mu0_te, mu1_te, np.full_like(mu0_te, t, dtype=np.float32))
        logit_t = ((mu_t - mu_center) / mu_scale * 1.0).astype(np.float32)
        p_grid_te.append(sigmoid_np(logit_t).astype(np.float32))
    p_grid_te = np.stack(p_grid_te, axis=1)
    p_grid_te = np.clip(p_grid_te, 1e-6, 1 - 1e-6)
    return dict(Xtr=Xtr, Ttr=Ttr, Ytr=Ytr, Xte=Xte, Tte=Tte, mu_grid_te=p_grid_te.astype(np.float32))


# ==========================================================
# Models: shared building blocks
# ==========================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128), out_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def poly_basis_t(t, deg):
    # t: (b,) -> (b, deg+1)
    feats = [torch.ones_like(t)]
    for k in range(1, deg + 1):
        feats.append(t ** k)
    return torch.stack(feats, dim=1)


# --------------------------
# TARNet
#   - binT: two heads (t=0/t=1)
#   - contT: single head taking [h, phi(t)]
# --------------------------
class TARNet(nn.Module):
    def __init__(self, x_dim, rep_dim=128, hidden=(128, 128), dropout=0.1,
                 tfeat="direct", n_freq=10, tmlp_dim=32, is_binary_t=False, is_binary_y=False):
        super().__init__()
        self.is_binary_t = is_binary_t
        self.is_binary_y = is_binary_y
        self.tfeat = tfeat
        self.n_freq = int(n_freq)

        self.trunk = MLP(x_dim, hidden=(256, 256), out_dim=rep_dim, dropout=dropout)

        # t feature dim
        if tfeat == "direct":
            self.tdim = 1
            self.tmlp = None
        elif tfeat == "fourier":
            self.tdim = 1 + 2 * self.n_freq
            self.tmlp = None
        elif tfeat == "mlp":
            self.tdim = int(tmlp_dim)
            self.tmlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, self.tdim),
            )
        else:
            raise ValueError("tfeat must be direct/fourier/mlp")

        if is_binary_t:
            # standard TARNet: two heads (no need to feed t)
            self.head0 = MLP(rep_dim, hidden=hidden, out_dim=1, dropout=dropout)
            self.head1 = MLP(rep_dim, hidden=hidden, out_dim=1, dropout=dropout)
        else:
            # continuous t: one head uses [h, tf(t)]
            self.head = MLP(rep_dim + self.tdim, hidden=hidden, out_dim=1, dropout=dropout)

    def t_features(self, t):
        # t: (b,)
        if self.tfeat == "direct":
            return t.unsqueeze(1)
        if self.tfeat == "mlp":
            return self.tmlp(t.unsqueeze(1))
        # fourier
        x = (t / 2.0) * math.pi
        feats = [torch.ones_like(x)]
        for k in range(1, self.n_freq + 1):
            feats.append(torch.sin(k * x))
            feats.append(torch.cos(k * x))
        return torch.stack(feats, dim=1)

    def forward(self, x, t):
        h = self.trunk(x)
        if self.is_binary_t:
            # choose head by t (0/1)
            y0 = self.head0(h).squeeze(1)
            y1 = self.head1(h).squeeze(1)
            return y0, y1
        tf = self.t_features(t)
        y = self.head(torch.cat([h, tf], dim=1)).squeeze(1)
        return y


# --------------------------
# DRNet (Schwab-style)
#   - contT: E heads by dose strata; each head sees [h, t]
#   - binT: E=2 acts like two-head TARNet
# --------------------------
class DRNet(nn.Module):
    def __init__(self, x_dim, rep_dim=128, E=10, hidden=(128, 128), dropout=0.1,
                 is_binary_t=False, is_binary_y=False):
        super().__init__()
        self.E = int(E)
        self.is_binary_t = is_binary_t
        self.is_binary_y = is_binary_y
        self.trunk = MLP(x_dim, hidden=(256, 256), out_dim=rep_dim, dropout=dropout)
        # each head takes [h, t]
        self.heads = nn.ModuleList([MLP(rep_dim + 1, hidden=hidden, out_dim=1, dropout=dropout) for _ in range(self.E)])

    def strata_id(self, t, t_min, t_max):
        # t: (b,)
        if self.is_binary_t:
            sid = t.long().clamp(0, 1)
            return sid
        u = (t - t_min) / (t_max - t_min + 1e-8)
        sid = torch.clamp((u * self.E).long(), 0, self.E - 1)
        return sid

    def forward(self, x, t, t_min, t_max):
        h = self.trunk(x)
        sid = self.strata_id(t, t_min, t_max)  # (b,)
        out = torch.zeros((x.size(0),), device=x.device)

        for e in range(self.E):
            idx = (sid == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            he = h[idx]
            te = t[idx].unsqueeze(1)
            ye = self.heads[e](torch.cat([he, te], dim=1)).squeeze(1)
            out[idx] = ye
        return out


# --------------------------
# VCNet-style (varying coefficient): mu(x,t)=a(x)^T phi(t)
# --------------------------
class VCNetStyle(nn.Module):
    def __init__(self, x_dim, deg=6, hidden=(128, 128), dropout=0.1, is_binary_y=False):
        super().__init__()
        self.deg = int(deg)
        self.is_binary_y = is_binary_y
        self.a_net = MLP(x_dim, hidden=(256, 256), out_dim=self.deg + 1, dropout=dropout)

    def forward(self, x, t):
        a = self.a_net(x)                  # (b, deg+1)
        phi = poly_basis_t(t, self.deg)    # (b, deg+1)
        y = torch.sum(a * phi, dim=1)      # (b,)
        return y


# ==========================================================
# Train / Predict
# ==========================================================
def split_fit_val(X, T, Y, seed, val_frac=0.2):
    idx = np.arange(X.shape[0])
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(idx)
    n_val = max(64, int(val_frac * len(idx)))
    va = idx[:n_val]
    tr = idx[n_val:]
    return (X[tr], T[tr], Y[tr]), (X[va], T[va], Y[va])

def train_earlystop_generic(model, train_batcher, Xva_t, Tva_t, Yva_t,
                            loss_fn, opt, epochs, patience, min_delta=1e-5):
    best = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for batch in train_batcher:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v = loss_fn(model, (Xva_t, Tva_t, Yva_t)).item()

        if v < best - min_delta:
            best = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_grid_cont(model, X, t_grid, device, kind, t_min=None, t_max=None):
    """
    Return mu_hat (n,G).
    kind: "tarnet" / "drnet" / "vcnet"
    """
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    out = []
    for tv in t_grid:
        t = torch.full((X.shape[0],), float(tv), device=device, dtype=torch.float32)
        if kind == "tarnet":
            y = model(X_t, t)
        elif kind == "vcnet":
            y = model(X_t, t)
        elif kind == "drnet":
            assert t_min is not None and t_max is not None
            y = model(X_t, t, t_min, t_max)
        else:
            raise ValueError(kind)
        out.append(y.detach().cpu().numpy().astype(np.float32))
    return np.stack(out, axis=1)

@torch.no_grad()
def predict_binT_mu01(model, X, device, mode, is_binary_y):
    """
    mode: "tarnet" OR "drnet2"  (both treated as two-head by construction)
    Return (mu0, mu1) as:
      - continuous Y: raw
      - binary Y: probability
    """
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    t_dummy = torch.zeros((X.shape[0],), device=device, dtype=torch.float32)

    if mode == "tarnet":
        y0, y1 = model(X_t, t_dummy)
    elif mode == "drnet2":
        # DRNet with E=2: call forward with t (0/1) separately to get each head behavior
        # easiest: directly feed t=0 and t=1 and let strata_id route
        t0 = torch.zeros((X.shape[0],), device=device, dtype=torch.float32)
        t1 = torch.ones((X.shape[0],), device=device, dtype=torch.float32)
        y0 = model(X_t, t0, 0.0, 1.0)
        y1 = model(X_t, t1, 0.0, 1.0)
    else:
        raise ValueError(mode)

    y0 = y0.detach().cpu().numpy().astype(np.float32)
    y1 = y1.detach().cpu().numpy().astype(np.float32)

    if is_binary_y:
        p0 = sigmoid_np(y0).astype(np.float32)
        p1 = sigmoid_np(y1).astype(np.float32)
        return np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)
    return y0, y1


# ==========================================================
# Main experiment
# ==========================================================
def run_one_method(task, method_name, model_ctor, Xtr, Ttr, Ytr, Xte, cfg, seed, conf_strength):
    """
    Train on train split with internal val; predict:
      - contT: grid curve (n,G)
      - binT: mu0/mu1 (n,)
    """
    device = cfg["device"]
    epochs = cfg["epochs"]
    patience = cfg["patience"]
    batch_size = cfg["batch"]
    lr = cfg["lr"]
    wd = cfg["wd"]

    is_contT = task.startswith("contT_")
    is_binT = task.startswith("binT_")
    is_binY = task.endswith("_binY")

    (X_fit, T_fit, Y_fit), (X_va, T_va, Y_va) = split_fit_val(Xtr, Ttr, Ytr, seed=seed, val_frac=0.2)

    X_fit_t = torch.from_numpy(X_fit.astype(np.float32)).to(device)
    T_fit_t = torch.from_numpy(T_fit.astype(np.float32)).to(device)
    Y_fit_t = torch.from_numpy(Y_fit.astype(np.float32)).to(device)

    X_va_t = torch.from_numpy(X_va.astype(np.float32)).to(device)
    T_va_t = torch.from_numpy(T_va.astype(np.float32)).to(device)
    Y_va_t = torch.from_numpy(Y_va.astype(np.float32)).to(device)

    ds = TensorDataset(X_fit_t, T_fit_t, Y_fit_t)
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    model = model_ctor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # loss function wrapper
    if is_contT:
        if is_binY:
            def loss_fn(m, batch):
                xb, tb, yb = batch
                pred = m(xb, tb)  # logit
                return F.binary_cross_entropy_with_logits(pred, yb)
        else:
            def loss_fn(m, batch):
                xb, tb, yb = batch
                pred = m(xb, tb)
                return F.mse_loss(pred, yb)
    else:
        # binT tasks: model gives mu0/mu1; choose factual head by T for training loss
        if is_binY:
            def loss_fn(m, batch):
                xb, tb, yb = batch
                # TARNet: returns (y0,y1)
                if method_name.lower().startswith("drnet"):
                    # DRNet E=2 forward returns scalar routed by t (need call with provided t)
                    pred = m(xb, tb, 0.0, 1.0)
                else:
                    y0, y1 = m(xb, tb)
                    pred = torch.where(tb < 0.5, y0, y1)
                return F.binary_cross_entropy_with_logits(pred, yb)
        else:
            def loss_fn(m, batch):
                xb, tb, yb = batch
                if method_name.lower().startswith("drnet"):
                    pred = m(xb, tb, 0.0, 1.0)
                else:
                    y0, y1 = m(xb, tb)
                    pred = torch.where(tb < 0.5, y0, y1)
                return F.mse_loss(pred, yb)

    model = train_earlystop_generic(
        model=model,
        train_batcher=dl,
        Xva_t=X_va_t, Tva_t=T_va_t, Yva_t=Y_va_t,
        loss_fn=loss_fn,
        opt=opt,
        epochs=epochs,
        patience=patience,
        min_delta=1e-5
    )

    # predict
    if is_contT:
        kind = "vcnet" if "VCNet" in method_name else ("drnet" if "DRNet" in method_name else "tarnet")
        mu_hat = predict_grid_cont(
            model, Xte, cfg["t_grid"], device,
            kind=kind, t_min=cfg["t_min"], t_max=cfg["t_max"]
        )
        if is_binY:
            mu_hat = np.clip(mu_hat, -30, 30)  # stability as logits
            mu_hat = sigmoid_np(mu_hat).astype(np.float32)
            mu_hat = np.clip(mu_hat, 1e-6, 1 - 1e-6)
        return mu_hat

    # binT: return (mu0, mu1)
    if "DRNet" in method_name:
        mu0, mu1 = predict_binT_mu01(model, Xte, device, mode="drnet2", is_binary_y=is_binY)
    else:
        mu0, mu1 = predict_binT_mu01(model, Xte, device, mode="tarnet", is_binary_y=is_binY)
    return np.stack([mu0, mu1], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="News dataset file (.npz or .csv) containing covariates X.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tasks", type=str, default="all4", choices=["all4","contT_contY","contT_binY","binT_contY","binT_binY"])
    ap.add_argument("--conf_list", type=str, default="0.1,1.0,5.0")
    ap.add_argument("--rep_n", type=int, default=20, help="How many reps (random splits) to run if data has no rep dimension.")
    ap.add_argument("--seeds", type=str, default="100,101,102")
    ap.add_argument("--test_ratio", type=float, default=0.1, help="Used only if dataset provides a single X (no x_train/x_test).")
    ap.add_argument("--label_col", type=str, default=None, help="CSV only: a column name to drop (label column).")

    # model & train hyperparams
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)

    # contT eval grid
    ap.add_argument("--t_min", type=float, default=-2.0)
    ap.add_argument("--t_max", type=float, default=2.0)
    ap.add_argument("--t_grid_size", type=int, default=50)

    # baseline knobs
    ap.add_argument("--drnet_E", type=int, default=10)
    ap.add_argument("--vc_deg", type=int, default=6)
    ap.add_argument("--tfeat", type=str, default="direct", choices=["direct","fourier","mlp"])  # for TARNet contT input
    ap.add_argument("--n_freq", type=int, default=10)
    ap.add_argument("--tmlp_dim", type=int, default=32)

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    conf_list = [float(x) for x in args.conf_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    if args.tasks == "all4":
        tasks = ["contT_contY","contT_binY","binT_contY","binT_binY"]
    else:
        tasks = [args.tasks]

    # FAST override
    if args.fast or (os.environ.get("FAST", "0") == "1"):
        args.epochs = min(args.epochs, 60)
        args.patience = min(args.patience, 8)
        args.batch = min(args.batch, 256)
        args.lr = max(args.lr, 2e-3)

    device = torch.device(args.device)

    t_grid = np.linspace(args.t_min, args.t_max, args.t_grid_size).astype(np.float32)
    cfg = dict(
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        t_grid=t_grid,
    )

    print("==================================================")
    print("DATA:", args.data_path)
    print("OUT :", args.out_dir)
    print("TASKS:", tasks)
    print("CONF:", conf_list, "| seeds:", seeds, "| rep_n:", args.rep_n)
    print("DEVICE:", device)
    print("FAST:", args.fast or (os.environ.get("FAST","0")=="1"))
    print("==================================================")

    # baseline constructors per task (binary/continuous T & Y)
    def build_methods(x_dim, task):
        is_binT = task.startswith("binT_")
        is_binY = task.endswith("_binY")
        methods = []

        # DRNet
        if is_binT:
            # use E=2 for binT
            def ctor_drnet():
                return DRNet(x_dim=x_dim, rep_dim=128, E=2, hidden=(128,128), dropout=0.1,
                             is_binary_t=True, is_binary_y=is_binY)
        else:
            def ctor_drnet():
                return DRNet(x_dim=x_dim, rep_dim=128, E=args.drnet_E, hidden=(128,128), dropout=0.1,
                             is_binary_t=False, is_binary_y=is_binY)
        methods.append(("DRNet", ctor_drnet))

        # VCNet-style
        def ctor_vcnet():
            return VCNetStyle(x_dim=x_dim, deg=args.vc_deg, hidden=(128,128), dropout=0.1, is_binary_y=is_binY)
        methods.append(("VCNet", ctor_vcnet))

        # TARNet
        def ctor_tarnet():
            return TARNet(
                x_dim=x_dim, rep_dim=128, hidden=(128,128), dropout=0.1,
                tfeat=args.tfeat, n_freq=args.n_freq, tmlp_dim=args.tmlp_dim,
                is_binary_t=is_binT, is_binary_y=is_binY
            )
        methods.append(("TARNet", ctor_tarnet))

        return methods

    # run
    for task in tasks:
        task_dir = os.path.join(args.out_dir, f"TASK_{task}")
        ensure_dir(task_dir)
        raw_csv = os.path.join(task_dir, f"raw_{task}.csv")
        sum_csv = os.path.join(task_dir, f"summary_{task}.csv")

        done, wall_hist = load_done_keys(raw_csv)
        print(f"\n========== TASK={task} | resume rows={len(done)} ==========")

        # reps: if dataset has rep dim, rep_id indexes it; else rep_id controls random split
        rep_ids = list(range(int(args.rep_n)))

        last_print = time.time()
        t_start = time.time()

        for conf_strength in conf_list:
            for rep_id in rep_ids:
                # load X (standardized) per rep
                # (rep_id/seed controls split if no predefined split)
                # We'll use "seed=0" for the split itself; actual DGP uses each run's seed.
                Xtr, Xte = load_news_X(args.data_path, rep_id=rep_id, seed=0, test_ratio=args.test_ratio, label_col=args.label_col)
                x_dim = Xtr.shape[1]
                methods = build_methods(x_dim, task)

                for seed in seeds:
                    # build semi-synthetic data for this (task, conf, rep, seed)
                    dgp = dgp_build(task, Xtr, Xte, conf_strength=conf_strength, seed=seed,
                                    t_min=args.t_min, t_max=args.t_max, t_grid=t_grid)
                    Xtr_i, Ttr_i, Ytr_i = dgp["Xtr"], dgp["Ttr"], dgp["Ytr"]
                    Xte_i = dgp["Xte"]

                    for mname, ctor in methods:
                        method = f"{mname}"
                        key = (task, float(conf_strength), int(rep_id), int(seed), method)
                        if key in done:
                            continue

                        set_all_seeds(seed + 123 + rep_id * 7)

                        t0 = time.time()
                        pred = run_one_method(task, method, ctor, Xtr_i, Ttr_i, Ytr_i, Xte_i,
                                              cfg=cfg, seed=seed, conf_strength=conf_strength)
                        wall = float(time.time() - t0)

                        row = dict(
                            task=task,
                            conf_strength=float(conf_strength),
                            rep_id=int(rep_id),
                            seed=int(seed),
                            method=method,
                            wall_time_sec=wall,
                        )

                        # metrics
                        if task.startswith("contT_"):
                            mu_true = dgp["mu_grid_te"]
                            row["mise"] = mise_grid(pred, mu_true, t_grid)
                            row["rmse_grid"] = rmse(pred, mu_true)
                        else:
                            mu0_true = dgp["mu0_te"]
                            mu1_true = dgp["mu1_te"]
                            ite_true = (mu1_true - mu0_true).astype(np.float32)

                            mu0_hat = pred[:, 0]
                            mu1_hat = pred[:, 1]
                            ite_hat = (mu1_hat - mu0_hat).astype(np.float32)

                            row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                            row["ate_err"] = abs(float(np.mean(ite_hat)) - float(np.mean(ite_true)))

                        append_row(raw_csv, row)
                        done.add(key)
                        wall_hist.append(wall)

                        if time.time() - last_print > 60:
                            df_tmp = pd.read_csv(raw_csv, engine="python", on_bad_lines="skip")
                            print(f"[Progress] rows={len(df_tmp)} | elapsed={(time.time()-t_start)/3600:.2f}h | last={task}/{conf_strength}/rep{rep_id}/seed{seed}/{method} wall={wall:.1f}s")
                            last_print = time.time()

        # summary
        df = pd.read_csv(raw_csv, engine="python", on_bad_lines="skip")
        if task.startswith("contT_"):
            summ = df.groupby(["conf_strength","method"])[["mise","rmse_grid"]].agg(["mean","std","count"]).reset_index()
        else:
            summ = df.groupby(["conf_strength","method"])[["sqrt_pehe","ate_err"]].agg(["mean","std","count"]).reset_index()
        summ.to_csv(sum_csv, index=False)
        print("Saved:", raw_csv)
        print("Saved:", sum_csv)

    print("\n✅ All done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
