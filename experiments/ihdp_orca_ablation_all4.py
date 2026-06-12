# -*- coding: utf-8 -*-
# ==========================================================
# IHDP (SERVER) - ALL 4 tasks in one script:
#   1) contT_contY
#   2) contT_binY
#   3) binT_contY
#   4) binT_binY
#
# For each task:
#   - ORCA ablations: 12 = nuisance(4) x tfeat(3)
#   - NoOrth: 3 = tfeat only (direct/fourier/mlp)
#
# conf_strength: [0.1, 1.0, 5.0]
# Robust resume: per-task raw csv, skip bad lines.
# ASCII-only comments to avoid encoding issues.
# ==========================================================

import os, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)

# --------------------------
# 1) Config
# --------------------------
FAST = (os.environ.get("FAST", "0") == "1")

IHDP_TRAIN = os.environ.get("IHDP_TRAIN", "/nas/zrp_data/ihdp_train.npz")
IHDP_TEST  = os.environ.get("IHDP_TEST",  "/nas/zrp_data/ihdp_test.npz")

OUT_ROOT = os.environ.get("OUT_ROOT", "/nas/zrp_data/ihdp_orca_ablations_server")
RUN_TAG  = os.environ.get("RUN_TAG", time.strftime("%Y%m%d_%H%M%S"))
OUT_DIR  = os.path.join(OUT_ROOT, f"IHDP_ORCA_NOORTH_ALL4_{RUN_TAG}_{'FAST' if FAST else 'FULL'}")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKS = ["contT_contY", "contT_binY", "binT_contY", "binT_binY"]
CONF_LIST = [0.1, 1.0, 5.0]

REP_N = int(os.environ.get("REP_N", "8" if FAST else "100"))
REP_N = max(1, min(REP_N, 100))
REP_IDS = list(range(REP_N))

TRAIN_SEEDS = [100, 101, 102]

NUIS_KINDS  = ["rf", "ridge", "mlp", "gbdt"]
TFEAT_KINDS = ["fourier", "direct", "mlp"]
NOORTH_TFEATS = ["fourier", "direct", "mlp"]

# contT range/grid
T_MIN, T_MAX = -2.0, 2.0
T_GRID_SIZE = int(os.environ.get("T_GRID_SIZE", "50"))
T_GRID = np.linspace(T_MIN, T_MAX, T_GRID_SIZE).astype(np.float32)

# ORCA net training
EPOCHS_MAX = 50 if FAST else 160
PATIENCE   = 6  if FAST else 15
MIN_DELTA  = 1e-4
BATCH      = 256 if FAST else 512
LR         = 2e-3 if FAST else 1e-3
WD         = 1e-5
HIDDEN     = 256
REP_DIM    = 128
DROPOUT    = 0.1
N_FREQ     = 10
TMLP_DIM   = 32
N_FOLDS    = 2

# nuisance params
RF_TREES   = 200 if FAST else 300
RF_NJOBS   = 8
GBDT_ITERS = 200 if FAST else 400

# nuisance torch-mlp
NUIS_MLP_H = 128
NUIS_MLP_EPOCHS_MAX = 100 if FAST else 220
NUIS_MLP_PATIENCE   = 10  if FAST else 20
NUIS_MLP_LR = 2e-3
NUIS_MLP_WD = 1e-5
NUIS_MLP_BATCH = 256

print("==================================================")
print("DEVICE:", DEVICE)
print("OUT_DIR:", OUT_DIR)
print("FAST:", FAST)
print("IHDP_TRAIN:", IHDP_TRAIN)
print("IHDP_TEST :", IHDP_TEST)
print("TASKS:", TASKS)
print("REP_N:", REP_N, "| TRAIN_SEEDS:", TRAIN_SEEDS, "| CONF_LIST:", CONF_LIST)
print("NUIS:", NUIS_KINDS, "| TFEAT:", TFEAT_KINDS, "| NoOrth:", NOORTH_TFEATS)
print("==================================================")

# --------------------------
# 2) Utils
# --------------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit_np(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)

def sqrt_pehe(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def rmse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mise_grid(mu_hat, mu_true):
    # (n,G)
    dt = float(T_GRID[1] - T_GRID[0])
    mu_hat = np.asarray(mu_hat, dtype=np.float32)
    mu_true = np.asarray(mu_true, dtype=np.float32)
    return float(np.mean(np.sum((mu_hat - mu_true) ** 2, axis=1) * dt))

def latex_escape(s: str) -> str:
    s = str(s)
    return (s.replace("\\", "\\textbackslash ")
             .replace("_", "\\_").replace("%","\\%").replace("&","\\&")
             .replace("#","\\#").replace("{","\\{").replace("}","\\}")
             .replace("^","\\textasciicircum ").replace("~","\\textasciitilde "))

def bootstrap_ci(values, n_boot=3000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    n = len(v)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = v[idx].mean()
    lo = float(np.quantile(boot, alpha/2))
    hi = float(np.quantile(boot, 1 - alpha/2))
    return float(v.mean()), lo, hi

def paired_delta(df_sub, method_a, method_b, metric, n_boot=3000, seed=0):
    wide = df_sub.pivot_table(
        index=["conf_strength","rep_id","train_seed"],
        columns="method",
        values=metric,
        aggfunc="first"
    ).dropna()
    if (method_a not in wide.columns) or (method_b not in wide.columns):
        return None
    delta = (wide[method_a] - wide[method_b]).values
    mean, lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    return {"n_pairs": int(len(delta)), "delta_mean": mean, "ci95_low": lo, "ci95_high": hi}

def load_done_keys(raw_csv):
    if not os.path.exists(raw_csv):
        return set(), []
    df = pd.read_csv(raw_csv, engine="python", on_bad_lines="skip")
    keys = set(zip(df.conf_strength, df.rep_id, df.train_seed, df.method))
    wall = df["wall_time_sec"].to_numpy(dtype=float) if "wall_time_sec" in df.columns else np.array([], dtype=float)
    return keys, wall.tolist()

def append_row(raw_csv, row: dict):
    df1 = pd.DataFrame([row])
    if not os.path.exists(raw_csv):
        df1.to_csv(raw_csv, index=False)
    else:
        df1.to_csv(raw_csv, mode="a", header=False, index=False)

# --------------------------
# 3) IHDP loader
# --------------------------
def load_ihdp_arrays():
    tr = np.load(IHDP_TRAIN)
    te = np.load(IHDP_TEST)
    # x: (n,d,rep), mu0/mu1: (n,rep), yf: (n,rep)
    Xtr_all = tr["x"].astype(np.float32)  # (672,25,100)
    Xte_all = te["x"].astype(np.float32)  # (75,25,100)
    mu0_tr_all = tr["mu0"].astype(np.float32)
    mu1_tr_all = tr["mu1"].astype(np.float32)
    mu0_te_all = te["mu0"].astype(np.float32)
    mu1_te_all = te["mu1"].astype(np.float32)
    yf_tr_all  = tr["yf"].astype(np.float32)   # observed factual outcome in original IHDP
    return Xtr_all, Xte_all, mu0_tr_all, mu1_tr_all, mu0_te_all, mu1_te_all, yf_tr_all

def standardize_x_per_rep(Xtr, Xte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    return Xtr_s, Xte_s

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

def mu_to_prob(mu0, mu1):
    # per-rep standardize then sigmoid
    m = float(np.mean(np.concatenate([mu0, mu1], axis=0)))
    s = float(np.std(np.concatenate([mu0, mu1], axis=0)) + 1e-6)
    z0 = (mu0 - m) / s
    z1 = (mu1 - m) / s
    p0 = sigmoid_np(z0).astype(np.float32)
    p1 = sigmoid_np(z1).astype(np.float32)
    return np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)

# Smooth mixing function g(t) in [0,1] for contT tasks
def g_of_t(t):
    # map [-2,2] -> [0,1] smoothly; slope can be tuned if you want harder tasks
    return sigmoid_np(1.25 * t).astype(np.float32)

def sample_bernoulli(p, rng):
    return (rng.uniform(0, 1, size=p.shape[0]) < p).astype(np.float32)

# --------------------------
# 4) Build per-task data from IHDP
# --------------------------
def build_task_data(task, Xtr_all, Xte_all, mu0_tr_all, mu1_tr_all, mu0_te_all, mu1_te_all, yf_tr_all,
                    rep_id, conf_strength, seed):
    # select rep
    Xtr = Xtr_all[:, :, rep_id].astype(np.float32)
    Xte = Xte_all[:, :, rep_id].astype(np.float32)
    Xtr_s, Xte_s = standardize_x_per_rep(Xtr, Xte)

    mu0_tr = mu0_tr_all[:, rep_id].astype(np.float32)
    mu1_tr = mu1_tr_all[:, rep_id].astype(np.float32)
    mu0_te = mu0_te_all[:, rep_id].astype(np.float32)
    mu1_te = mu1_te_all[:, rep_id].astype(np.float32)

    rng = np.random.default_rng(seed + 999 + int(conf_strength * 1000) + rep_id * 7)

    if task == "binT_contY":
        # resample treatment by conf_strength
        e_tr = make_propensity_from_x(Xtr_s, conf_strength, seed=seed + 1234 + rep_id * 17)
        e_te = make_propensity_from_x(Xte_s, conf_strength, seed=seed + 5678 + rep_id * 29)
        Ttr = (rng.uniform(0, 1, size=Xtr_s.shape[0]) < e_tr).astype(np.float32)
        Tte = (rng.uniform(0, 1, size=Xte_s.shape[0]) < e_te).astype(np.float32)

        # generate Ytr from mu0/mu1 + noise
        Ytr = np.where(Ttr < 0.5, mu0_tr, mu1_tr).astype(np.float32) + rng.normal(0, 1.0, size=Xtr_s.shape[0]).astype(np.float32)

        # ground truth ITE on test
        ite_true = (mu1_te - mu0_te).astype(np.float32)
        ate_true = float(np.mean(ite_true))
        return dict(
            Xtr=Xtr_s, Ttr=Ttr, Ytr=Ytr,
            Xte=Xte_s, Tte=Tte,
            ite_true=ite_true, ate_true=ate_true,
            mu0_te=mu0_te.astype(np.float32), mu1_te=mu1_te.astype(np.float32)
        )

    if task == "binT_binY":
        p0_tr, p1_tr = mu_to_prob(mu0_tr, mu1_tr)
        p0_te, p1_te = mu_to_prob(mu0_te, mu1_te)

        e_tr = make_propensity_from_x(Xtr_s, conf_strength, seed=seed + 1234 + rep_id * 17)
        e_te = make_propensity_from_x(Xte_s, conf_strength, seed=seed + 5678 + rep_id * 29)
        Ttr = (rng.uniform(0, 1, size=Xtr_s.shape[0]) < e_tr).astype(np.float32)
        Tte = (rng.uniform(0, 1, size=Xte_s.shape[0]) < e_te).astype(np.float32)

        Ytr = np.empty((Xtr_s.shape[0],), dtype=np.float32)
        Ytr[Ttr < 0.5] = sample_bernoulli(p0_tr[Ttr < 0.5], rng)
        Ytr[Ttr >= 0.5] = sample_bernoulli(p1_tr[Ttr >= 0.5], rng)

        ite_true = (p1_te - p0_te).astype(np.float32)
        ate_true = float(np.mean(ite_true))
        return dict(
            Xtr=Xtr_s, Ttr=Ttr, Ytr=Ytr,
            Xte=Xte_s, Tte=Tte,
            ite_true=ite_true, ate_true=ate_true,
            mu0_te=p0_te.astype(np.float32), mu1_te=p1_te.astype(np.float32)
        )

    if task == "contT_contY":
        # generate continuous treatment
        score_tr = make_score_from_x(Xtr_s, seed=seed + 111 + rep_id * 3)
        score_te = make_score_from_x(Xte_s, seed=seed + 222 + rep_id * 5)
        Ttr = np.clip(conf_strength * score_tr + rng.normal(0, 1.0, size=Xtr_s.shape[0]), T_MIN, T_MAX).astype(np.float32)
        Tte = np.clip(conf_strength * score_te + rng.normal(0, 1.0, size=Xte_s.shape[0]), T_MIN, T_MAX).astype(np.float32)

        # build dose-response from mu0/mu1 using smooth mix g(t)
        # mu(x,t) = mu0 + g(t)*(mu1-mu0)
        def mu_cont(mu0, mu1, t):
            g = g_of_t(t)
            return (mu0 + g * (mu1 - mu0)).astype(np.float32)

        mu_f_tr = mu_cont(mu0_tr, mu1_tr, Ttr)
        Ytr = mu_f_tr + rng.normal(0, 1.0, size=Xtr_s.shape[0]).astype(np.float32)

        mu_grid_te = np.stack([mu_cont(mu0_te, mu1_te, np.full_like(mu0_te, t, dtype=np.float32)) for t in T_GRID], axis=1)
        return dict(
            Xtr=Xtr_s, Ttr=Ttr, Ytr=Ytr,
            Xte=Xte_s, Tte=Tte,
            mu_grid_te=mu_grid_te.astype(np.float32)
        )

    if task == "contT_binY":
        p0_tr, p1_tr = mu_to_prob(mu0_tr, mu1_tr)
        p0_te, p1_te = mu_to_prob(mu0_te, mu1_te)

        score_tr = make_score_from_x(Xtr_s, seed=seed + 111 + rep_id * 3)
        score_te = make_score_from_x(Xte_s, seed=seed + 222 + rep_id * 5)
        Ttr = np.clip(conf_strength * score_tr + rng.normal(0, 1.0, size=Xtr_s.shape[0]), T_MIN, T_MAX).astype(np.float32)
        Tte = np.clip(conf_strength * score_te + rng.normal(0, 1.0, size=Xte_s.shape[0]), T_MIN, T_MAX).astype(np.float32)

        def p_cont(p0, p1, t):
            g = g_of_t(t)
            return np.clip(p0 + g * (p1 - p0), 1e-6, 1 - 1e-6).astype(np.float32)

        p_f_tr = p_cont(p0_tr, p1_tr, Ttr)
        Ytr = sample_bernoulli(p_f_tr, rng)

        p_grid_te = np.stack([p_cont(p0_te, p1_te, np.full_like(p0_te, t, dtype=np.float32)) for t in T_GRID], axis=1)
        return dict(
            Xtr=Xtr_s, Ttr=Ttr, Ytr=Ytr,
            Xte=Xte_s, Tte=Tte,
            mu_grid_te=p_grid_te.astype(np.float32)  # treat as "truth grid" for MISE
        )

    raise ValueError("Unknown task: " + task)

# --------------------------
# 5) Nuisance models
# --------------------------
class TorchMLPRegressor:
    def __init__(self, seed: int):
        self.seed = seed
        self.scaler = StandardScaler()
        self.net = None

    def fit(self, X, y):
        set_all_seeds(self.seed)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)
        n = Xs.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed + 77)
        rng.shuffle(idx)
        n_val = max(64, int(0.2 * n))
        va = idx[:n_val]; tr = idx[n_val:]
        Xtr, ytr = Xs[tr], y[tr]
        Xva, yva = Xs[va], y[va]

        ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        dl = DataLoader(ds, batch_size=min(NUIS_MLP_BATCH, len(Xtr)), shuffle=True)

        net = nn.Sequential(
            nn.Linear(Xs.shape[1], NUIS_MLP_H), nn.ReLU(),
            nn.Linear(NUIS_MLP_H, NUIS_MLP_H), nn.ReLU(),
            nn.Linear(NUIS_MLP_H, 1),
        ).to(DEVICE)

        opt = torch.optim.Adam(net.parameters(), lr=NUIS_MLP_LR, weight_decay=NUIS_MLP_WD)
        mse = nn.MSELoss()

        Xva_t = torch.from_numpy(Xva).to(DEVICE)
        yva_t = torch.from_numpy(yva).to(DEVICE)

        best = float("inf"); best_state=None; wait=0
        for _ in range(NUIS_MLP_EPOCHS_MAX):
            net.train()
            for xb, yb in dl:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                pred = net(xb).squeeze()
                loss = mse(pred, yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = mse(net(Xva_t).squeeze(), yva_t).item()
            if v < best - 1e-5:
                best = v
                best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= NUIS_MLP_PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    @torch.no_grad()
    def predict(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        xb = torch.from_numpy(Xs).to(DEVICE)
        self.net.eval()
        return self.net(xb).squeeze().detach().cpu().numpy().astype(np.float32)

class TorchMLPClassifier:
    def __init__(self, seed: int):
        self.seed = seed
        self.scaler = StandardScaler()
        self.net = None

    def fit(self, X, y01):
        set_all_seeds(self.seed)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        y = y01.astype(np.float32)
        n = Xs.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed + 99)
        rng.shuffle(idx)
        n_val = max(64, int(0.2 * n))
        va = idx[:n_val]; tr = idx[n_val:]
        Xtr, ytr = Xs[tr], y[tr]
        Xva, yva = Xs[va], y[va]

        ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        dl = DataLoader(ds, batch_size=min(NUIS_MLP_BATCH, len(Xtr)), shuffle=True)

        net = nn.Sequential(
            nn.Linear(Xs.shape[1], NUIS_MLP_H), nn.ReLU(),
            nn.Linear(NUIS_MLP_H, NUIS_MLP_H), nn.ReLU(),
            nn.Linear(NUIS_MLP_H, 1),
        ).to(DEVICE)

        opt = torch.optim.Adam(net.parameters(), lr=NUIS_MLP_LR, weight_decay=NUIS_MLP_WD)
        bce = nn.BCEWithLogitsLoss()

        Xva_t = torch.from_numpy(Xva).to(DEVICE)
        yva_t = torch.from_numpy(yva).to(DEVICE)

        best = float("inf"); best_state=None; wait=0
        for _ in range(NUIS_MLP_EPOCHS_MAX):
            net.train()
            for xb, yb in dl:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                logit = net(xb).squeeze()
                loss = bce(logit, yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = bce(net(Xva_t).squeeze(), yva_t).item()
            if v < best - 1e-5:
                best = v
                best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= NUIS_MLP_PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        xb = torch.from_numpy(Xs).to(DEVICE)
        self.net.eval()
        logit = self.net(xb).squeeze().detach().cpu().numpy().astype(np.float32)
        p = sigmoid_np(logit)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.stack([1 - p, p], axis=1)

def make_nuisance_reg(kind: str, seed: int):
    k = kind.lower()
    if k == "rf":
        return RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=10, random_state=seed, n_jobs=RF_NJOBS)
    if k == "ridge":
        return Ridge(alpha=1.0)
    if k == "gbdt":
        return HistGradientBoostingRegressor(max_iter=GBDT_ITERS, learning_rate=0.05, random_state=seed)
    if k == "mlp":
        return TorchMLPRegressor(seed)
    raise ValueError(k)

def make_nuisance_clf(kind: str, seed: int):
    k = kind.lower()
    if k == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=10, random_state=seed, n_jobs=RF_NJOBS)
    if k == "ridge":
        return LogisticRegression(max_iter=2000)
    if k == "gbdt":
        return HistGradientBoostingClassifier(max_iter=GBDT_ITERS, learning_rate=0.05, random_state=seed)
    if k == "mlp":
        return TorchMLPClassifier(seed)
    raise ValueError(k)

# --------------------------
# 6) ORCA model
# --------------------------
class ResidualNetFlex(nn.Module):
    def __init__(self, input_dim, rep_dim, hidden, dropout, tfeat_kind="fourier", n_freq=10, tmlp_dim=32):
        super().__init__()
        self.tfeat_kind = tfeat_kind
        self.n_freq = int(n_freq)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep_dim),
        )

        if self.tfeat_kind == "fourier":
            self.basis_dim = 1 + 2 * self.n_freq
            self.tmlp = None
        elif self.tfeat_kind == "direct":
            self.basis_dim = 1
            self.tmlp = None
        elif self.tfeat_kind == "mlp":
            self.basis_dim = int(tmlp_dim)
            self.tmlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, self.basis_dim),
            )
        else:
            raise ValueError("tfeat_kind must be fourier/direct/mlp")

        self.head = nn.Sequential(
            nn.Linear(rep_dim + self.basis_dim + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def t_features(self, t):
        if self.tfeat_kind == "direct":
            return t.unsqueeze(1)
        if self.tfeat_kind == "mlp":
            return self.tmlp(t.unsqueeze(1))
        x = (t / 2.0) * math.pi
        feats = [torch.ones_like(x)]
        for k in range(1, self.n_freq + 1):
            feats.append(torch.sin(k * x))
            feats.append(torch.cos(k * x))
        return torch.stack(feats, dim=1)

    def forward(self, x, t, ttil):
        rep = self.encoder(x)
        tf = self.t_features(t)
        return self.head(torch.cat([rep, tf, ttil.unsqueeze(1)], dim=1)).squeeze()

def _split_train_val(n, seed, val_frac=0.1, min_val=64):
    idx = np.arange(n)
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(idx)
    n_val = max(min_val, int(val_frac * n))
    va = idx[:n_val]
    tr = idx[n_val:]
    return tr, va

def train_contY_net(X, T, y_target, ttil, seed, tfeat_kind):
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=64)

    Xtr, Ttr, Ytr, TTtr = X[tr_idx], T[tr_idx], y_target[tr_idx], ttil[tr_idx]
    Xva, Tva, Yva, TTva = X[va_idx], T[va_idx], y_target[va_idx], ttil[va_idx]

    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(Ttr.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
        torch.from_numpy(TTtr.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=min(BATCH, len(Xtr)), shuffle=True, drop_last=False, pin_memory=True)

    xb_va = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    tb_va = torch.from_numpy(Tva.astype(np.float32)).to(DEVICE)
    yb_va = torch.from_numpy(Yva.astype(np.float32)).to(DEVICE)
    tt_va = torch.from_numpy(TTva.astype(np.float32)).to(DEVICE)

    net = ResidualNetFlex(
        input_dim=X.shape[1], rep_dim=REP_DIM, hidden=HIDDEN, dropout=DROPOUT,
        tfeat_kind=tfeat_kind, n_freq=N_FREQ, tmlp_dim=TMLP_DIM
    ).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    mse = nn.MSELoss()

    best = float("inf"); best_state=None; wait=0
    for _ in range(EPOCHS_MAX):
        net.train()
        for xb, tb, yb, ttb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            ttb = ttb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = net(xb, tb, ttb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            v = mse(net(xb_va, tb_va, tt_va), yb_va).item()

        if v < best - MIN_DELTA:
            best = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net

def train_binY_net_with_offset(X, T, y01, ttil, base_logit, seed, tfeat_kind):
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=64)

    Xtr, Ttr, Ytr, TTtr, Ltr = X[tr_idx], T[tr_idx], y01[tr_idx], ttil[tr_idx], base_logit[tr_idx]
    Xva, Tva, Yva, TTva, Lva = X[va_idx], T[va_idx], y01[va_idx], ttil[va_idx], base_logit[va_idx]

    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(Ttr.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
        torch.from_numpy(TTtr.astype(np.float32)),
        torch.from_numpy(Ltr.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=min(BATCH, len(Xtr)), shuffle=True, drop_last=False, pin_memory=True)

    xb_va = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    tb_va = torch.from_numpy(Tva.astype(np.float32)).to(DEVICE)
    yb_va = torch.from_numpy(Yva.astype(np.float32)).to(DEVICE)
    tt_va = torch.from_numpy(TTva.astype(np.float32)).to(DEVICE)
    lb_va = torch.from_numpy(Lva.astype(np.float32)).to(DEVICE)

    net = ResidualNetFlex(
        input_dim=X.shape[1], rep_dim=REP_DIM, hidden=HIDDEN, dropout=DROPOUT,
        tfeat_kind=tfeat_kind, n_freq=N_FREQ, tmlp_dim=TMLP_DIM
    ).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()

    best = float("inf"); best_state=None; wait=0
    for _ in range(EPOCHS_MAX):
        net.train()
        for xb, tb, yb, ttb, lb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            ttb = ttb.to(DEVICE, non_blocking=True)
            lb = lb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            delta = net(xb, tb, ttb)
            logits = lb + delta
            loss = bce(logits, yb)
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            v = bce(lb_va + net(xb_va, tb_va, tt_va), yb_va).item()

        if v < best - MIN_DELTA:
            best = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net

# ORCA predictors
@torch.no_grad()
def orca_predict_contT_contY(net, m_model, e_model, X, t_values):
    # return mu_hat on grid: (n, G)
    m_hat = m_model.predict(X).astype(np.float32)
    # e(X) for contT: regression -> predicted T
    e_hat = e_model.predict(X).astype(np.float32)
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)

    out = []
    for t in t_values:
        tt = np.full((X.shape[0],), float(t), dtype=np.float32)
        ttil = (tt - e_hat).astype(np.float32)
        ytil = net(x_t,
                   torch.from_numpy(tt).to(DEVICE),
                   torch.from_numpy(ttil).to(DEVICE)
                   ).cpu().numpy().astype(np.float32)
        out.append((m_hat + ytil).astype(np.float32))
    return np.stack(out, axis=1)

@torch.no_grad()
def orca_predict_contT_binY(net, m_model, e_model, X, t_values):
    # m_model predicts prob -> base_logit
    p_hat = m_model.predict_proba(X)[:, 1].astype(np.float32)
    base_logit = logit_np(p_hat)
    e_hat = e_model.predict(X).astype(np.float32)  # predicted T (regression)
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)

    out = []
    for t in t_values:
        tt = np.full((X.shape[0],), float(t), dtype=np.float32)
        ttil = (tt - e_hat).astype(np.float32)
        delta = net(x_t,
                    torch.from_numpy(tt).to(DEVICE),
                    torch.from_numpy(ttil).to(DEVICE)
                    ).cpu().numpy().astype(np.float32)
        prob = sigmoid_np(base_logit + delta).astype(np.float32)
        out.append(np.clip(prob, 1e-6, 1 - 1e-6))
    return np.stack(out, axis=1)

@torch.no_grad()
def orca_predict_binT_contY(net, m_model, e_model, X):
    m_hat = m_model.predict(X).astype(np.float32)
    e_hat = e_model.predict_proba(X)[:, 1].astype(np.float32)
    e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)

    t0 = np.zeros((X.shape[0],), dtype=np.float32)
    t1 = np.ones((X.shape[0],), dtype=np.float32)

    y0 = net(x_t,
             torch.from_numpy(t0).to(DEVICE),
             torch.from_numpy((t0 - e_hat).astype(np.float32)).to(DEVICE)
             ).cpu().numpy().astype(np.float32)
    y1 = net(x_t,
             torch.from_numpy(t1).to(DEVICE),
             torch.from_numpy((t1 - e_hat).astype(np.float32)).to(DEVICE)
             ).cpu().numpy().astype(np.float32)
    return (m_hat + y0).astype(np.float32), (m_hat + y1).astype(np.float32)

@torch.no_grad()
def orca_predict_binT_binY(net, m_model, e_model, X):
    p_hat = m_model.predict_proba(X)[:, 1].astype(np.float32)
    base_logit = logit_np(p_hat)
    e_hat = e_model.predict_proba(X)[:, 1].astype(np.float32)
    e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)

    t0 = np.zeros((X.shape[0],), dtype=np.float32)
    t1 = np.ones((X.shape[0],), dtype=np.float32)

    d0 = net(x_t,
             torch.from_numpy(t0).to(DEVICE),
             torch.from_numpy((t0 - e_hat).astype(np.float32)).to(DEVICE)
             ).cpu().numpy().astype(np.float32)
    d1 = net(x_t,
             torch.from_numpy(t1).to(DEVICE),
             torch.from_numpy((t1 - e_hat).astype(np.float32)).to(DEVICE)
             ).cpu().numpy().astype(np.float32)
    p0 = sigmoid_np(base_logit + d0).astype(np.float32)
    p1 = sigmoid_np(base_logit + d1).astype(np.float32)
    return np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)

def orca_fit_predict(task, Xtr, Ttr, Ytr, Xte, seed, nuisance_kind, tfeat_kind):
    # cross-fitting on train, predict on test
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    preds = []

    for fold_id, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
        X_fit, T_fit, Y_fit = Xtr[idx_fit], Ttr[idx_fit], Ytr[idx_fit]
        X_res, T_res, Y_res = Xtr[idx_res], Ttr[idx_res], Ytr[idx_res]

        if task.startswith("contT_"):
            # e(X): regression on T
            e_model = make_nuisance_reg(nuisance_kind, seed + 20 + fold_id).fit(X_fit, T_fit)
            e_hat = e_model.predict(X_res).astype(np.float32)
            # m(X): outcome model
            if task.endswith("_contY"):
                m_model = make_nuisance_reg(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit)
                m_hat = m_model.predict(X_res).astype(np.float32)
                y_target = (Y_res.astype(np.float32) - m_hat).astype(np.float32)
                ttil = (T_res.astype(np.float32) - e_hat).astype(np.float32)
                net = train_contY_net(X_res, T_res, y_target, ttil, seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)
                mu_grid_hat = orca_predict_contT_contY(net, m_model, e_model, Xte, T_GRID)
                preds.append(mu_grid_hat)
            else:
                m_model = make_nuisance_clf(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit.astype(int))
                p_res = m_model.predict_proba(X_res)[:, 1].astype(np.float32)
                base_logit_res = logit_np(p_res)
                y_target = (Y_res.astype(np.float32) - p_res).astype(np.float32)  # residual in prob space
                ttil = (T_res.astype(np.float32) - e_hat).astype(np.float32)
                net = train_binY_net_with_offset(X_res, T_res, Y_res.astype(np.float32), ttil, base_logit_res,
                                                 seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)
                p_grid_hat = orca_predict_contT_binY(net, m_model, e_model, Xte, T_GRID)
                preds.append(p_grid_hat)

        else:
            # binT tasks
            e_model = make_nuisance_clf(nuisance_kind, seed + 20 + fold_id).fit(X_fit, T_fit.astype(int))
            e_hat = e_model.predict_proba(X_res)[:, 1].astype(np.float32)
            e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

            if task.endswith("_contY"):
                m_model = make_nuisance_reg(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit)
                m_hat = m_model.predict(X_res).astype(np.float32)
                y_target = (Y_res.astype(np.float32) - m_hat).astype(np.float32)
                ttil = (T_res.astype(np.float32) - e_hat).astype(np.float32)
                net = train_contY_net(X_res, T_res, y_target, ttil, seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)
                mu0_hat, mu1_hat = orca_predict_binT_contY(net, m_model, e_model, Xte)
                preds.append(np.stack([mu0_hat, mu1_hat], axis=1))
            else:
                m_model = make_nuisance_clf(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit.astype(int))
                p_res = m_model.predict_proba(X_res)[:, 1].astype(np.float32)
                base_logit_res = logit_np(p_res)
                y_target = (Y_res.astype(np.float32) - p_res).astype(np.float32)
                ttil = (T_res.astype(np.float32) - e_hat).astype(np.float32)
                net = train_binY_net_with_offset(X_res, T_res, Y_res.astype(np.float32), ttil, base_logit_res,
                                                 seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)
                p0_hat, p1_hat = orca_predict_binT_binY(net, m_model, e_model, Xte)
                preds.append(np.stack([p0_hat, p1_hat], axis=1))

    return np.mean(np.stack(preds, axis=0), axis=0)

# --------------------------
# 7) NoOrth baselines
# --------------------------
class NoOrthNet(nn.Module):
    # direct: concat [rep, t] ; fourier/mlp similar to ResidualNetFlex but without ttil
    def __init__(self, input_dim, rep_dim, hidden, dropout, tfeat_kind="fourier", n_freq=10, tmlp_dim=32):
        super().__init__()
        self.tfeat_kind = tfeat_kind
        self.n_freq = int(n_freq)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep_dim),
        )

        if self.tfeat_kind == "fourier":
            self.basis_dim = 1 + 2 * self.n_freq
            self.tmlp = None
        elif self.tfeat_kind == "direct":
            self.basis_dim = 1
            self.tmlp = None
        elif self.tfeat_kind == "mlp":
            self.basis_dim = int(tmlp_dim)
            self.tmlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, self.basis_dim),
            )
        else:
            raise ValueError("tfeat_kind must be fourier/direct/mlp")

        self.head = nn.Sequential(
            nn.Linear(rep_dim + self.basis_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def t_features(self, t):
        if self.tfeat_kind == "direct":
            return t.unsqueeze(1)
        if self.tfeat_kind == "mlp":
            return self.tmlp(t.unsqueeze(1))
        x = (t / 2.0) * math.pi
        feats = [torch.ones_like(x)]
        for k in range(1, self.n_freq + 1):
            feats.append(torch.sin(k * x))
            feats.append(torch.cos(k * x))
        return torch.stack(feats, dim=1)

    def forward(self, x, t):
        rep = self.encoder(x)
        tf = self.t_features(t)
        return self.head(torch.cat([rep, tf], dim=1)).squeeze()

def train_noorth_contY(X, T, Y, seed, tfeat_kind):
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=64)
    Xtr, Ttr, Ytr = X[tr_idx], T[tr_idx], Y[tr_idx]
    Xva, Tva, Yva = X[va_idx], T[va_idx], Y[va_idx]

    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(Ttr.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=min(BATCH, len(Xtr)), shuffle=True, drop_last=False, pin_memory=True)

    xb_va = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    tb_va = torch.from_numpy(Tva.astype(np.float32)).to(DEVICE)
    yb_va = torch.from_numpy(Yva.astype(np.float32)).to(DEVICE)

    net = NoOrthNet(
        input_dim=X.shape[1], rep_dim=REP_DIM, hidden=HIDDEN, dropout=DROPOUT,
        tfeat_kind=tfeat_kind, n_freq=N_FREQ, tmlp_dim=TMLP_DIM
    ).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    mse = nn.MSELoss()

    best = float("inf"); best_state=None; wait=0
    for _ in range(EPOCHS_MAX):
        net.train()
        for xb, tb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = net(xb, tb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = mse(net(xb_va, tb_va), yb_va).item()
        if v < best - MIN_DELTA:
            best = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net

def train_noorth_binY(X, T, Y01, seed, tfeat_kind):
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=64)
    Xtr, Ttr, Ytr = X[tr_idx], T[tr_idx], Y01[tr_idx]
    Xva, Tva, Yva = X[va_idx], T[va_idx], Y01[va_idx]

    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(Ttr.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=min(BATCH, len(Xtr)), shuffle=True, drop_last=False, pin_memory=True)

    xb_va = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    tb_va = torch.from_numpy(Tva.astype(np.float32)).to(DEVICE)
    yb_va = torch.from_numpy(Yva.astype(np.float32)).to(DEVICE)

    net = NoOrthNet(
        input_dim=X.shape[1], rep_dim=REP_DIM, hidden=HIDDEN, dropout=DROPOUT,
        tfeat_kind=tfeat_kind, n_freq=N_FREQ, tmlp_dim=TMLP_DIM
    ).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()

    best = float("inf"); best_state=None; wait=0
    for _ in range(EPOCHS_MAX):
        net.train()
        for xb, tb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logit = net(xb, tb)
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = bce(net(xb_va, tb_va), yb_va).item()
        if v < best - MIN_DELTA:
            best = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net

@torch.no_grad()
def noorth_predict_contT_contY(net, X, t_values):
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    out = []
    for t in t_values:
        tt = np.full((X.shape[0],), float(t), dtype=np.float32)
        y = net(x_t, torch.from_numpy(tt).to(DEVICE)).cpu().numpy().astype(np.float32)
        out.append(y)
    return np.stack(out, axis=1)

@torch.no_grad()
def noorth_predict_contT_binY(net, X, t_values):
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    out = []
    for t in t_values:
        tt = np.full((X.shape[0],), float(t), dtype=np.float32)
        logit = net(x_t, torch.from_numpy(tt).to(DEVICE)).cpu().numpy().astype(np.float32)
        p = sigmoid_np(logit).astype(np.float32)
        out.append(np.clip(p, 1e-6, 1 - 1e-6))
    return np.stack(out, axis=1)

@torch.no_grad()
def noorth_predict_binT(net, X):
    x_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    t0 = np.zeros((X.shape[0],), dtype=np.float32)
    t1 = np.ones((X.shape[0],), dtype=np.float32)
    y0 = net(x_t, torch.from_numpy(t0).to(DEVICE)).cpu().numpy().astype(np.float32)
    y1 = net(x_t, torch.from_numpy(t1).to(DEVICE)).cpu().numpy().astype(np.float32)
    return y0, y1

# --------------------------
# 8) Postprocess (per task)
# --------------------------
def make_heatmap(df_task, metric, out_dir, task, conf_strength, nuis_order, tfeat_order):
    sub = df_task[df_task["conf_strength"] == conf_strength].copy()
    if len(sub) == 0:
        return

    def parse_method(m):
        inside = m[m.find("[")+1:m.rfind("]")]
        # ORCA[RF|t=fourier] OR NoOrth[t=fourier]
        if m.startswith("ORCA["):
            nuis, tpart = inside.split("|")
            tfeat = tpart.split("=")[1]
            return nuis.lower(), tfeat.lower()
        else:
            # NoOrth[t=xxx] -> nuisance blank
            tfeat = inside.split("=")[1]
            return "noorth", tfeat.lower()

    sub["nuis2"], sub["tfeat2"] = zip(*sub["method"].map(parse_method))

    mat = sub.groupby(["nuis2","tfeat2"])[metric].mean().reset_index()
    if "noorth" not in nuis_order:
        nuis_order2 = nuis_order + ["noorth"]
    else:
        nuis_order2 = nuis_order

    piv = mat.pivot(index="nuis2", columns="tfeat2", values=metric).reindex(index=nuis_order2, columns=tfeat_order)

    plt.figure(figsize=(7.0, 4.8))
    plt.imshow(piv.values, aspect="auto")
    plt.xticks(range(len(piv.columns)), piv.columns)
    plt.yticks(range(len(piv.index)), piv.index)
    plt.colorbar(label=f"mean {metric}")
    plt.title(f"{task} heatmap (conf={conf_strength})")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"heatmap_{task}_conf_{conf_strength}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)

def write_topk_latex(summary_df, metric, out_tex, caption, label, topk=8):
    tmp = summary_df.copy()
    # expected columns after groupby-agg reset:
    # conf_strength, method, metric_mean, metric_std, metric_count, ...
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(f"conf & Method & {metric} (mean$\\pm$sd) & $n$ \\\\")
    lines.append("\\midrule")

    for cs in sorted(tmp.conf_strength.unique()):
        sub = tmp[tmp.conf_strength == cs].sort_values(f"{metric}_mean").head(topk)
        first = True
        for _, r in sub.iterrows():
            conf_cell = f"{cs:g}" if first else ""
            sd = 0.0 if np.isnan(r[f"{metric}_std"]) else float(r[f"{metric}_std"])
            cell = f"{float(r[f'{metric}_mean']):.4g}$\\pm${sd:.3g}"
            lines.append(f"{conf_cell} & {latex_escape(r['method'])} & {cell} & {int(r[f'{metric}_count'])} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote:", out_tex)

# --------------------------
# 9) Main loop (all tasks)
# --------------------------
def main():
    Xtr_all, Xte_all, mu0_tr_all, mu1_tr_all, mu0_te_all, mu1_te_all, yf_tr_all = load_ihdp_arrays()

    for task in TASKS:
        task_dir = os.path.join(OUT_DIR, f"TASK_{task}")
        os.makedirs(task_dir, exist_ok=True)

        RAW_CSV  = os.path.join(task_dir, f"raw_{task}.csv")
        SUM_CSV  = os.path.join(task_dir, f"summary_{task}.csv")
        CI_CSV   = os.path.join(task_dir, f"paired_bootstrap_ci_{task}.csv")
        TOPK_TEX = os.path.join(task_dir, f"table_topk_{task}.tex")

        done, wall_hist = load_done_keys(RAW_CSV)
        print(f"\n========== TASK={task} | resume rows={len(done)} ==========")

        planned = len(CONF_LIST) * len(REP_IDS) * len(TRAIN_SEEDS) * (len(NUIS_KINDS) * len(TFEAT_KINDS) + len(NOORTH_TFEATS))
        print("Planned rows (ORCA 12 + NoOrth 3):", planned)

        t0 = time.time()
        last_print = time.time()

        if task.startswith("contT_"):
            metric_primary = "mise"
            metric_aux = "rmse_grid"
        else:
            metric_primary = "sqrt_pehe"
            metric_aux = "ate_err"

        for cs in CONF_LIST:
            for rep_id in REP_IDS:
                for seed in TRAIN_SEEDS:
                    data = build_task_data(
                        task, Xtr_all, Xte_all, mu0_tr_all, mu1_tr_all, mu0_te_all, mu1_te_all, yf_tr_all,
                        rep_id=rep_id, conf_strength=cs, seed=seed
                    )
                    Xtr, Ttr, Ytr = data["Xtr"], data["Ttr"], data["Ytr"]
                    Xte = data["Xte"]

                    # --------------------------
                    # (A) ORCA ablations: nuisance x tfeat
                    # --------------------------
                    for nuis in NUIS_KINDS:
                        for tfeat in TFEAT_KINDS:
                            method = f"ORCA[{nuis.upper()}|t={tfeat}]"
                            key = (float(cs), int(rep_id), int(seed), method)
                            if key in done:
                                continue

                            start = time.time()
                            pred = orca_fit_predict(
                                task, Xtr, Ttr, Ytr, Xte,
                                seed=seed, nuisance_kind=nuis, tfeat_kind=tfeat
                            )

                            row = dict(
                                task=task, conf_strength=float(cs), rep_id=int(rep_id), train_seed=int(seed),
                                method=method, nuisance=nuis, tfeat=tfeat, orth=True,
                                wall_time_sec=float(time.time() - start),
                            )

                            if task == "contT_contY":
                                mu_true = data["mu_grid_te"]
                                row["mise"] = mise_grid(pred, mu_true)
                                row["rmse_grid"] = rmse(pred, mu_true)

                            elif task == "contT_binY":
                                p_true = data["mu_grid_te"]
                                row["mise"] = mise_grid(pred, p_true)
                                row["rmse_grid"] = rmse(pred, p_true)

                            elif task == "binT_contY":
                                mu0_hat = pred[:, 0]; mu1_hat = pred[:, 1]
                                ite_hat = (mu1_hat - mu0_hat).astype(np.float32)
                                ite_true = data["ite_true"]
                                row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                                ate_hat = float(np.mean(ite_hat))
                                row["ate_err"] = abs(ate_hat - float(np.mean(ite_true)))

                            elif task == "binT_binY":
                                p0_hat = pred[:, 0]; p1_hat = pred[:, 1]
                                ite_hat = (p1_hat - p0_hat).astype(np.float32)
                                ite_true = data["ite_true"]
                                row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                                ate_hat = float(np.mean(ite_hat))
                                row["ate_err"] = abs(ate_hat - float(np.mean(ite_true)))

                            append_row(RAW_CSV, row)
                            done.add(key)
                            wall_hist.append(float(row["wall_time_sec"]))

                            if time.time() - last_print > 60:
                                total_done = len(done)
                                elapsed = time.time() - t0
                                if len(wall_hist) >= 10:
                                    med = float(np.median(wall_hist[-min(300, len(wall_hist)):] ))
                                    remaining = planned - total_done
                                    eta_sec = remaining * med
                                    print(f"[Progress] {total_done}/{planned} | elapsed={elapsed/3600:.2f}h | median_row={med:.2f}s | ETA~{eta_sec/3600:.2f}h")
                                else:
                                    print(f"[Progress] {total_done}/{planned} | elapsed={elapsed/3600:.2f}h")
                                last_print = time.time()

                    # --------------------------
                    # (B) NoOrth ablations: only tfeat
                    # --------------------------
                    for tfeat in NOORTH_TFEATS:
                        method = f"NoOrth[t={tfeat}]"
                        key = (float(cs), int(rep_id), int(seed), method)
                        if key in done:
                            continue

                        start = time.time()

                        if task.endswith("_contY"):
                            net = train_noorth_contY(Xtr, Ttr, Ytr, seed=seed + 300, tfeat_kind=tfeat)
                            if task.startswith("contT_"):
                                pred = noorth_predict_contT_contY(net, Xte, T_GRID)  # (n,G)
                            else:
                                y0, y1 = noorth_predict_binT(net, Xte)
                                pred = np.stack([y0, y1], axis=1)  # (n,2)
                        else:
                            net = train_noorth_binY(Xtr, Ttr, Ytr.astype(np.float32), seed=seed + 300, tfeat_kind=tfeat)
                            if task.startswith("contT_"):
                                pred = noorth_predict_contT_binY(net, Xte, T_GRID)  # (n,G) prob
                            else:
                                l0, l1 = noorth_predict_binT(net, Xte)
                                p0 = sigmoid_np(l0).astype(np.float32)
                                p1 = sigmoid_np(l1).astype(np.float32)
                                pred = np.stack([np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)], axis=1)

                        row = dict(
                            task=task, conf_strength=float(cs), rep_id=int(rep_id), train_seed=int(seed),
                            method=method, nuisance="noorth", tfeat=tfeat, orth=False,
                            wall_time_sec=float(time.time() - start),
                        )

                        if task == "contT_contY":
                            mu_true = data["mu_grid_te"]
                            row["mise"] = mise_grid(pred, mu_true)
                            row["rmse_grid"] = rmse(pred, mu_true)

                        elif task == "contT_binY":
                            p_true = data["mu_grid_te"]
                            row["mise"] = mise_grid(pred, p_true)
                            row["rmse_grid"] = rmse(pred, p_true)

                        elif task == "binT_contY":
                            mu0_hat = pred[:, 0]; mu1_hat = pred[:, 1]
                            ite_hat = (mu1_hat - mu0_hat).astype(np.float32)
                            ite_true = data["ite_true"]
                            row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                            ate_hat = float(np.mean(ite_hat))
                            row["ate_err"] = abs(ate_hat - float(np.mean(ite_true)))

                        elif task == "binT_binY":
                            p0_hat = pred[:, 0]; p1_hat = pred[:, 1]
                            ite_hat = (p1_hat - p0_hat).astype(np.float32)
                            ite_true = data["ite_true"]
                            row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                            ate_hat = float(np.mean(ite_hat))
                            row["ate_err"] = abs(ate_hat - float(np.mean(ite_true)))

                        append_row(RAW_CSV, row)
                        done.add(key)
                        wall_hist.append(float(row["wall_time_sec"]))

                        if time.time() - last_print > 60:
                            total_done = len(done)
                            elapsed = time.time() - t0
                            if len(wall_hist) >= 10:
                                med = float(np.median(wall_hist[-min(300, len(wall_hist)):] ))
                                remaining = planned - total_done
                                eta_sec = remaining * med
                                print(f"[Progress] {total_done}/{planned} | elapsed={elapsed/3600:.2f}h | median_row={med:.2f}s | ETA~{eta_sec/3600:.2f}h")
                            else:
                                print(f"[Progress] {total_done}/{planned} | elapsed={elapsed/3600:.2f}h")
                            last_print = time.time()

        print(f"Done runs for TASK={task}. RAW_CSV:", RAW_CSV)

        # --------------------------
        # Postprocess per task
        # --------------------------
        df = pd.read_csv(RAW_CSV, engine="python", on_bad_lines="skip")

        if task.startswith("contT_"):
            summ = df.groupby(["conf_strength","method"])[["mise","rmse_grid"]].agg(["mean","std","count"]).reset_index()
            summ.to_csv(SUM_CSV, index=False)
            print("Saved summary:", SUM_CSV)

            tmp = summ.copy()
            tmp.columns = [
                "conf_strength","method",
                "mise_mean","mise_std","mise_count",
                "rmse_grid_mean","rmse_grid_std","rmse_grid_count"
            ]
            write_topk_latex(
                tmp, "mise", TOPK_TEX,
                caption=f"IHDP ablations (TASK={latex_escape(task)}). Lower is better.",
                label=f"tab:ihdp_{task}_topk",
                topk=8
            )

            nuis_order = ["rf","ridge","mlp","gbdt","noorth"]
            tfeat_order = ["fourier","direct","mlp"]
            for cs in sorted(df["conf_strength"].unique()):
                make_heatmap(df, "mise", task_dir, task, cs, nuis_order, tfeat_order)

            # Paired bootstrap CI: best per conf vs others
            mean_tbl = df.groupby(["conf_strength","method"])["mise"].mean().reset_index()
            best_tbl = (mean_tbl.sort_values(["conf_strength","mise"])
                                .groupby("conf_strength")
                                .head(1)
                                .rename(columns={"method":"best_method"}))

            rows = []
            for cs in sorted(df["conf_strength"].unique()):
                best = best_tbl[best_tbl.conf_strength == cs].iloc[0]["best_method"]
                sub = df[df.conf_strength == cs].copy()
                methods = sorted(sub["method"].unique())
                for b in methods:
                    if b == best:
                        continue
                    res = paired_delta(sub, best, b, metric="mise", n_boot=3000, seed=int(float(cs)*1000) + 19)
                    if res is None:
                        continue
                    rows.append({
                        "conf_strength": float(cs),
                        "best": best,
                        "baseline": b,
                        "comparison": f"{best} - {b}",
                        **res
                    })
            pd.DataFrame(rows).to_csv(CI_CSV, index=False)
            print("Saved paired bootstrap CI:", CI_CSV)

        else:
            summ = df.groupby(["conf_strength","method"])[["sqrt_pehe","ate_err"]].agg(["mean","std","count"]).reset_index()
            summ.to_csv(SUM_CSV, index=False)
            print("Saved summary:", SUM_CSV)

            tmp = summ.copy()
            tmp.columns = [
                "conf_strength","method",
                "sqrt_pehe_mean","sqrt_pehe_std","sqrt_pehe_count",
                "ate_err_mean","ate_err_std","ate_err_count"
            ]
            write_topk_latex(
                tmp, "sqrt_pehe", TOPK_TEX,
                caption=f"IHDP ablations (TASK={latex_escape(task)}). Lower is better.",
                label=f"tab:ihdp_{task}_topk",
                topk=8
            )

            nuis_order = ["rf","ridge","mlp","gbdt","noorth"]
            tfeat_order = ["fourier","direct","mlp"]
            for cs in sorted(df["conf_strength"].unique()):
                make_heatmap(df, "sqrt_pehe", task_dir, task, cs, nuis_order, tfeat_order)

            # Paired bootstrap CI: best per conf vs others
            mean_tbl = df.groupby(["conf_strength","method"])["sqrt_pehe"].mean().reset_index()
            best_tbl = (mean_tbl.sort_values(["conf_strength","sqrt_pehe"])
                                .groupby("conf_strength")
                                .head(1)
                                .rename(columns={"method":"best_method"}))

            rows = []
            for cs in sorted(df["conf_strength"].unique()):
                best = best_tbl[best_tbl.conf_strength == cs].iloc[0]["best_method"]
                sub = df[df.conf_strength == cs].copy()
                methods = sorted(sub["method"].unique())
                for b in methods:
                    if b == best:
                        continue
                    res = paired_delta(sub, best, b, metric="sqrt_pehe", n_boot=3000, seed=int(float(cs)*1000) + 19)
                    if res is None:
                        continue
                    rows.append({
                        "conf_strength": float(cs),
                        "best": best,
                        "baseline": b,
                        "comparison": f"{best} - {b}",
                        **res
                    })
            pd.DataFrame(rows).to_csv(CI_CSV, index=False)
            print("Saved paired bootstrap CI:", CI_CSV)

        print("\nAll outputs saved in:", OUT_DIR)
        print("Task folder:", task_dir)
        print("Key files:", RAW_CSV, SUM_CSV, CI_CSV, TOPK_TEX)


if __name__ == "__main__":
    main()
