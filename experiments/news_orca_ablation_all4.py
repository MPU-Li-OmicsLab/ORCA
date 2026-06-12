# -*- coding: utf-8 -*-
# ==========================================================
# NEWS (SERVER) - ORCA ablations (12 = nuisance x tfeat) + NoOrth (3 = tfeat)
# Tasks:
#   - contT_contY
#   - contT_binY
#   - binT_contY
#   - binT_binY
# conf_strength: [0.1, 1.0, 5.0]
#
# Features:
#   - per-task raw csv (fixed schema) -> robust resume, skip bad lines
#   - per-task summary csv, topk latex, heatmap, paired bootstrap CI
#   - no pandas.read_csv inside training loop (ETA uses in-memory wall times)
# ==========================================================

import os, time, math, warnings, gzip, urllib.request
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)

# --------------------------
# 1) Config (env overridable)
# --------------------------
FAST = (os.environ.get("FAST", "0") == "1")

TASKS = ["contT_contY", "contT_binY", "binT_contY", "binT_binY"]

CONF_LIST = [0.1, 1.0, 5.0]
REP_IDS = list(range(int(os.environ.get("REP_N", "2" if FAST else "5"))))
TRAIN_SEEDS = [100, 101, 102]

# NYTimes BOW
N_DOCS = int(os.environ.get("N_DOCS", "3000" if FAST else "5000"))
TOP_VOCAB = int(os.environ.get("TOP_VOCAB", "800" if FAST else "2000"))

# Output
OUT_ROOT = os.environ.get("OUT_ROOT", "/nas/zrp_data/news_orca_ablations_server")
RUN_TAG  = os.environ.get("RUN_TAG", time.strftime("%Y%m%d_%H%M%S"))
OUT_DIR  = os.path.join(OUT_ROOT, f"NEWS_ORCA_NOORTH_ALL4_{RUN_TAG}_{'FAST' if FAST else 'FULL'}")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache dirs
CACHE_DIR = os.path.join(OUT_DIR, "_dgp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Use a fixed cache dir for BOW to avoid re-parsing per OUT_DIR
DATA_CACHE_DIR = os.environ.get("NEWS_DATA_DIR", "/nas/zrp_data/news_bow_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Treatment grid for contT tasks
T_MIN, T_MAX = -2.0, 2.0
T_GRID_SIZE = int(os.environ.get("T_GRID_SIZE", "50"))
T_GRID = np.linspace(T_MIN, T_MAX, T_GRID_SIZE).astype(np.float32)

# Ablations
NUIS_KINDS  = ["rf", "ridge", "mlp", "gbdt"]
TFEAT_KINDS = ["fourier", "direct", "mlp"]       # ORCA: 12 = 4 x 3
NOORTH_TFEATS = ["fourier", "direct", "mlp"]     # NoOrth: 3

# Train params
EPOCHS_MAX = 60 if FAST else 160
PATIENCE   = 8 if FAST else 15
MIN_DELTA  = 1e-4
BATCH      = 512 if FAST else 1024
LR         = 2e-3 if FAST else 1e-3
WD         = 1e-5
HIDDEN     = 256
REP_DIM    = 128
DROPOUT    = 0.1
N_FREQ     = 10
TMLP_DIM   = 32
N_FOLDS    = 2

# Nuisance params
RF_TREES   = 250 if FAST else 300
RF_NJOBS   = 8
GBDT_ITERS = 200 if FAST else 400

# Nuisance Torch-MLP
NUIS_MLP_H = 128
NUIS_MLP_EPOCHS_MAX = 120 if FAST else 250
NUIS_MLP_PATIENCE   = 12 if FAST else 25
NUIS_MLP_LR = 2e-3
NUIS_MLP_WD = 1e-5
NUIS_MLP_BATCH = 512

print("==================================================")
print("DEVICE:", DEVICE)
print("OUT_DIR:", OUT_DIR)
print("FAST:", FAST, "| N_DOCS:", N_DOCS, "| TOP_VOCAB:", TOP_VOCAB)
print("TASKS:", TASKS)
print("REP_IDS:", REP_IDS, "| TRAIN_SEEDS:", TRAIN_SEEDS, "| CONF_LIST:", CONF_LIST)
print("NUIS:", NUIS_KINDS, "| TFEAT:", TFEAT_KINDS, "| NoOrth TFEAT:", NOORTH_TFEATS)
print("==================================================")

# --------------------------
# 2) Helpers
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

def rmse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def sqrt_pehe(ite_hat, ite_true):
    return rmse(ite_hat, ite_true)

def mise_grid(mu_hat, mu_true):
    # mu_hat/mu_true: (n, G)
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

def cache_path(task, rep_id, conf_strength):
    tag = f"{task}_rep{rep_id}_conf{conf_strength}".replace(".", "p")
    return os.path.join(CACHE_DIR, f"{tag}.npz")

def bootstrap_ci(values, n_boot=4000, alpha=0.05, seed=0):
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

def paired_delta(df_sub, method_a, method_b, metric, n_boot=4000, seed=0):
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

# --------------------------
# 3) NEWS BOW download/parse (cached)
# --------------------------
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words"

def download_news_bow(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    doc_path = os.path.join(data_dir, "docword.nytimes.txt.gz")
    vocab_path = os.path.join(data_dir, "vocab.nytimes.txt")

    if not os.path.exists(doc_path):
        url_doc = f"{UCI_BASE_URL}/docword.nytimes.txt.gz"
        print("Downloading:", url_doc)
        urllib.request.urlretrieve(url_doc, doc_path)
        print("Saved:", doc_path)
    else:
        print("Found:", doc_path)

    if not os.path.exists(vocab_path):
        url_vocab = f"{UCI_BASE_URL}/vocab.nytimes.txt"
        print("Downloading:", url_vocab)
        urllib.request.urlretrieve(url_vocab, vocab_path)
        print("Saved:", vocab_path)
    else:
        print("Found:", vocab_path)

    return doc_path, vocab_path

def load_news_bow(n_docs, top_vocab, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_x = os.path.join(cache_dir, f"X_n{n_docs}_v{top_vocab}.npy")
    if os.path.exists(cache_x):
        X = np.load(cache_x).astype(np.float32)
        print("Loaded cached X:", cache_x, X.shape)
        return X

    doc_path, _ = download_news_bow(cache_dir)
    print("Parsing NYTimes docword file... (first time only)")

    with gzip.open(doc_path, "rt") as f:
        D = int(f.readline().strip())
        W = int(f.readline().strip())
        NNZ = int(f.readline().strip())
        print(f"Header: D={D}, W={W}, NNZ={NNZ}")
        n_docs = min(n_docs, D)
        print(f"Using first {n_docs} documents.")
        word_counts = np.zeros(W, dtype=np.int64)
        for line in f:
            doc_id, word_id, cnt = map(int, line.split())
            word_counts[word_id - 1] += cnt

    top_vocab = min(top_vocab, W)
    top_idx = np.argpartition(word_counts, -top_vocab)[-top_vocab:]
    top_idx = np.sort(top_idx)
    id2col = {int(i + 1): j for j, i in enumerate(top_idx)}
    print("Selected top", top_vocab, "words.")

    X = np.zeros((n_docs, top_vocab), dtype=np.float32)
    with gzip.open(doc_path, "rt") as f:
        _ = f.readline(); _ = f.readline(); _ = f.readline()
        for line in f:
            doc_id, word_id, cnt = map(int, line.split())
            if doc_id <= n_docs and word_id in id2col:
                j = id2col[word_id]
                X[doc_id - 1, j] += float(cnt)

    X = np.log1p(X).astype(np.float32)
    np.save(cache_x, X)
    print("Saved cached X:", cache_x, X.shape)
    return X

# --------------------------
# 4) Semi-synth DGP for 4 tasks (NEWS)
# --------------------------
def _make_score(Xz, rng, k=8):
    n, d = Xz.shape
    k = min(k, d)
    w = rng.normal(0, 1, size=(k,)).astype(np.float32)
    return (Xz[:, :k] @ w).astype(np.float32)

def _baseline_eff(Xz):
    x0 = Xz[:, 0]
    x1 = Xz[:, 1] if Xz.shape[1] > 1 else Xz[:, 0]
    baseline = 2.0 * (x0 ** 2) + x1 + 1.0

    def eff(t):
        return 2.0 * np.sin(np.pi * x0 * t) + 2.0 * (x1 * t) ** 2 + t ** 3
    return baseline.astype(np.float32), eff

def dgp_contT_contY(X_base, seed, conf_strength):
    rng = np.random.default_rng(seed)
    Xz = StandardScaler().fit_transform(X_base).astype(np.float32)

    baseline, eff = _baseline_eff(Xz)
    score = _make_score(Xz, rng, k=8)

    T = (conf_strength * score + rng.normal(0, 1.0, size=Xz.shape[0])).astype(np.float32)
    T = np.clip(T, T_MIN, T_MAX).astype(np.float32)

    mu_grid = np.stack([(baseline + eff(t)).astype(np.float32) for t in T_GRID], axis=1)
    mu_f = (baseline + eff(T)).astype(np.float32)
    Y = (mu_f + rng.normal(0, 1.0, size=Xz.shape[0])).astype(np.float32)
    return Xz, T, Y, mu_grid

def dgp_contT_binY(X_base, seed, conf_strength):
    rng = np.random.default_rng(seed)
    Xz = StandardScaler().fit_transform(X_base).astype(np.float32)

    baseline, eff = _baseline_eff(Xz)
    score = _make_score(Xz, rng, k=8)
    T = (conf_strength * score + rng.normal(0, 1.0, size=Xz.shape[0])).astype(np.float32)
    T = np.clip(T, T_MIN, T_MAX).astype(np.float32)

    scale = 2.5
    p_grid = np.stack([sigmoid_np((baseline + eff(t)) / scale).astype(np.float32) for t in T_GRID], axis=1)
    p_f = sigmoid_np((baseline + eff(T)) / scale).astype(np.float32)
    Y = (rng.uniform(0, 1, size=Xz.shape[0]) < p_f).astype(np.float32)
    return Xz, T, Y, p_grid

def dgp_binT_contY(X_base, seed, conf_strength):
    rng = np.random.default_rng(seed)
    Xz = StandardScaler().fit_transform(X_base).astype(np.float32)

    baseline, eff = _baseline_eff(Xz)
    score = _make_score(Xz, rng, k=8)
    e = sigmoid_np(conf_strength * score).astype(np.float32)
    e = np.clip(e, 1e-6, 1 - 1e-6)
    T = (rng.uniform(0, 1, size=Xz.shape[0]) < e).astype(np.float32)

    mu0 = (baseline + eff(0.0)).astype(np.float32)
    mu1 = (baseline + eff(1.0)).astype(np.float32)
    mu_f = (baseline + eff(T)).astype(np.float32)
    Y = (mu_f + rng.normal(0, 1.0, size=Xz.shape[0])).astype(np.float32)
    return Xz, T, Y, mu0, mu1

def dgp_binT_binY(X_base, seed, conf_strength):
    rng = np.random.default_rng(seed)
    Xz = StandardScaler().fit_transform(X_base).astype(np.float32)

    baseline, eff = _baseline_eff(Xz)
    score = _make_score(Xz, rng, k=8)
    e = sigmoid_np(conf_strength * score).astype(np.float32)
    e = np.clip(e, 1e-6, 1 - 1e-6)
    T = (rng.uniform(0, 1, size=Xz.shape[0]) < e).astype(np.float32)

    scale = 2.5
    p0 = sigmoid_np((baseline + eff(0.0)) / scale).astype(np.float32)
    p1 = sigmoid_np((baseline + eff(1.0)) / scale).astype(np.float32)
    pf = sigmoid_np((baseline + eff(T)) / scale).astype(np.float32)
    Y = (rng.uniform(0, 1, size=Xz.shape[0]) < pf).astype(np.float32)
    return Xz, T, Y, p0, p1

def get_split_cached_news(task, X_base, rep_id, conf_strength):
    cp = cache_path(task, rep_id, conf_strength)
    if os.path.exists(cp):
        z = np.load(cp)
        Xtr = z["Xtr"]; Ttr = z["Ttr"]; Ytr = z["Ytr"]
        Xte = z["Xte"]; Tte = z["Tte"]; Yte = z["Yte"]
        if task.startswith("contT_"):
            mu_grid_te = z["mu_grid_te"]
            return Xtr, Ttr, Ytr, Xte, Tte, Yte, mu_grid_te
        else:
            mu0_te = z["mu0_te"]; mu1_te = z["mu1_te"]
            return Xtr, Ttr, Ytr, Xte, Tte, Yte, mu0_te, mu1_te

    seed = 54321 + rep_id + int(conf_strength * 10)

    if task == "contT_contY":
        Xz, T, Y, mu_grid = dgp_contT_contY(X_base, seed=seed, conf_strength=conf_strength)
        Xtr, Xte, Ttr, Tte, Ytr, Yte, _, mu_grid_te = train_test_split(
            Xz, T, Y, mu_grid, test_size=0.2, random_state=999 + rep_id + int(conf_strength * 10)
        )
        np.savez_compressed(cp, Xtr=Xtr, Ttr=Ttr, Ytr=Ytr, Xte=Xte, Tte=Tte, Yte=Yte, mu_grid_te=mu_grid_te)
        return Xtr, Ttr, Ytr, Xte, Tte, Yte, mu_grid_te

    if task == "contT_binY":
        Xz, T, Y, p_grid = dgp_contT_binY(X_base, seed=seed, conf_strength=conf_strength)
        Xtr, Xte, Ttr, Tte, Ytr, Yte, _, p_grid_te = train_test_split(
            Xz, T, Y, p_grid, test_size=0.2, random_state=999 + rep_id + int(conf_strength * 10)
        )
        np.savez_compressed(cp, Xtr=Xtr, Ttr=Ttr, Ytr=Ytr, Xte=Xte, Tte=Tte, Yte=Yte, mu_grid_te=p_grid_te)
        return Xtr, Ttr, Ytr, Xte, Tte, Yte, p_grid_te

    if task == "binT_contY":
        Xz, T, Y, mu0, mu1 = dgp_binT_contY(X_base, seed=seed, conf_strength=conf_strength)
        Xtr, Xte, Ttr, Tte, Ytr, Yte, _, mu0_te, _, mu1_te = train_test_split(
            Xz, T, Y, mu0, mu1, test_size=0.2, random_state=999 + rep_id + int(conf_strength * 10)
        )
        np.savez_compressed(cp, Xtr=Xtr, Ttr=Ttr, Ytr=Ytr, Xte=Xte, Tte=Tte, Yte=Yte, mu0_te=mu0_te, mu1_te=mu1_te)
        return Xtr, Ttr, Ytr, Xte, Tte, Yte, mu0_te, mu1_te

    if task == "binT_binY":
        Xz, T, Y, p0, p1 = dgp_binT_binY(X_base, seed=seed, conf_strength=conf_strength)
        Xtr, Xte, Ttr, Tte, Ytr, Yte, _, p0_te, _, p1_te = train_test_split(
            Xz, T, Y, p0, p1, test_size=0.2, random_state=999 + rep_id + int(conf_strength * 10)
        )
        np.savez_compressed(cp, Xtr=Xtr, Ttr=Ttr, Ytr=Ytr, Xte=Xte, Tte=Tte, Yte=Yte, mu0_te=p0_te, mu1_te=p1_te)
        return Xtr, Ttr, Ytr, Xte, Tte, Yte, p0_te, p1_te

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
# 6) ResidualNetFlex
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

# --------------------------
# 7) Train helpers
# --------------------------
def _split_train_val(n, seed, val_frac=0.1, min_val=128):
    idx = np.arange(n)
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(idx)
    n_val = max(min_val, int(val_frac * n))
    va = idx[:n_val]
    tr = idx[n_val:]
    return tr, va

def train_contY_net(X, T, y_target, ttil, seed, tfeat_kind):
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=128)

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
    # net outputs delta-logit; final logit = base_logit + net(...)
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=128)

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

def train_binY_noorth_net(X, T, y01, seed, tfeat_kind):
    # NoOrth: logit = net(x,t,0); probability = sigmoid(logit)
    set_all_seeds(seed)
    tr_idx, va_idx = _split_train_val(X.shape[0], seed, val_frac=0.1, min_val=128)

    Xtr, Ttr, Ytr = X[tr_idx], T[tr_idx], y01[tr_idx]
    Xva, Tva, Yva = X[va_idx], T[va_idx], y01[va_idx]

    ttil_tr = np.zeros_like(Ttr, dtype=np.float32)
    ttil_va = np.zeros_like(Tva, dtype=np.float32)

    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(Ttr.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
        torch.from_numpy(ttil_tr.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=min(BATCH, len(Xtr)), shuffle=True, drop_last=False, pin_memory=True)

    xb_va = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    tb_va = torch.from_numpy(Tva.astype(np.float32)).to(DEVICE)
    yb_va = torch.from_numpy(Yva.astype(np.float32)).to(DEVICE)
    tt_va = torch.from_numpy(ttil_va.astype(np.float32)).to(DEVICE)

    net = ResidualNetFlex(
        input_dim=X.shape[1], rep_dim=REP_DIM, hidden=HIDDEN, dropout=DROPOUT,
        tfeat_kind=tfeat_kind, n_freq=N_FREQ, tmlp_dim=TMLP_DIM
    ).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()

    best = float("inf"); best_state=None; wait=0
    for _ in range(EPOCHS_MAX):
        net.train()
        for xb, tb, yb, ttb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            ttb = ttb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = net(xb, tb, ttb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            v = bce(net(xb_va, tb_va, tt_va), yb_va).item()

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

# --------------------------
# 8) ORCA + NoOrth runners
# --------------------------
@torch.no_grad()
def _predict_grid_cont(net, Xte, e_hat_te, tfeat_kind, use_offset_logit=False, base_logit=None):
    # returns (n, G) float32; if use_offset_logit: base_logit + net -> sigmoid
    x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)
    e_hat_te = e_hat_te.astype(np.float32)
    out = np.empty((Xte.shape[0], len(T_GRID)), dtype=np.float32)

    for j, tval in enumerate(T_GRID):
        t = np.full((Xte.shape[0],), tval, dtype=np.float32)
        ttil = (t - e_hat_te).astype(np.float32)
        tb = torch.from_numpy(t).to(DEVICE)
        ttb = torch.from_numpy(ttil).to(DEVICE)
        delta = net(x_t, tb, ttb).detach().cpu().numpy().astype(np.float32)
        if use_offset_logit:
            logits = base_logit + delta
            out[:, j] = sigmoid_np(logits).astype(np.float32)
        else:
            out[:, j] = delta  # caller adds m_hat for contY
    return out

def orca_contT_contY_mu_grid(Xtr, Ttr, Ytr, Xte, seed, nuisance_kind, tfeat_kind):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    preds = []

    for fold_id, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
        X_fit, T_fit, Y_fit = Xtr[idx_fit], Ttr[idx_fit], Ytr[idx_fit]
        X_res, T_res, Y_res = Xtr[idx_res], Ttr[idx_res], Ytr[idx_res]

        e_model = make_nuisance_reg(nuisance_kind, seed + 20 + fold_id).fit(X_fit, T_fit)
        e_hat_res = e_model.predict(X_res).astype(np.float32)
        e_hat_te  = e_model.predict(Xte).astype(np.float32)

        m_model = make_nuisance_reg(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit)
        m_hat_res = m_model.predict(X_res).astype(np.float32)
        m_hat_te  = m_model.predict(Xte).astype(np.float32)

        y_target = (Y_res.astype(np.float32) - m_hat_res).astype(np.float32)
        ttil = (T_res.astype(np.float32) - e_hat_res).astype(np.float32)

        net = train_contY_net(X_res, T_res, y_target, ttil, seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)

        delta_grid = _predict_grid_cont(net, Xte, e_hat_te, tfeat_kind, use_offset_logit=False)
        mu_grid = (m_hat_te[:, None] + delta_grid).astype(np.float32)
        preds.append(mu_grid)

    return np.mean(np.stack(preds, 0), 0)

def orca_contT_binY_p_grid(Xtr, Ttr, Ytr01, Xte, seed, nuisance_kind, tfeat_kind):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    preds = []

    for fold_id, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
        X_fit, T_fit, Y_fit = Xtr[idx_fit], Ttr[idx_fit], Ytr01[idx_fit]
        X_res, T_res, Y_res = Xtr[idx_res], Ttr[idx_res], Ytr01[idx_res]

        e_model = make_nuisance_reg(nuisance_kind, seed + 20 + fold_id).fit(X_fit, T_fit)
        e_hat_res = e_model.predict(X_res).astype(np.float32)
        e_hat_te  = e_model.predict(Xte).astype(np.float32)

        m_model = make_nuisance_clf(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit.astype(int))
        m_hat_res = m_model.predict_proba(X_res)[:, 1].astype(np.float32)
        m_hat_te  = m_model.predict_proba(Xte)[:, 1].astype(np.float32)

        base_logit_res = logit_np(m_hat_res)
        base_logit_te  = logit_np(m_hat_te)

        ttil = (T_res.astype(np.float32) - e_hat_res).astype(np.float32)
        net = train_binY_net_with_offset(
            X_res, T_res, Y_res.astype(np.float32), ttil, base_logit_res,
            seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind
        )

        # predict p-grid
        x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)
        out = np.empty((Xte.shape[0], len(T_GRID)), dtype=np.float32)
        for j, tval in enumerate(T_GRID):
            t = np.full((Xte.shape[0],), tval, dtype=np.float32)
            ttil_te = (t - e_hat_te).astype(np.float32)
            tb = torch.from_numpy(t).to(DEVICE)
            ttb = torch.from_numpy(ttil_te).to(DEVICE)
            delta = net(x_t, tb, ttb).detach().cpu().numpy().astype(np.float32)
            out[:, j] = sigmoid_np(base_logit_te + delta).astype(np.float32)
        preds.append(out)

    return np.mean(np.stack(preds, 0), 0)

def orca_binT_contY_mu01(Xtr, Ttr, Ytr, Xte, seed, nuisance_kind, tfeat_kind):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    mu0s, mu1s = [], []

    for fold_id, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
        X_fit, T_fit, Y_fit = Xtr[idx_fit], Ttr[idx_fit], Ytr[idx_fit]
        X_res, T_res, Y_res = Xtr[idx_res], Ttr[idx_res], Ytr[idx_res]

        e_model = make_nuisance_clf(nuisance_kind, seed + 20 + fold_id).fit(X_fit, T_fit.astype(int))
        e_hat_res = e_model.predict_proba(X_res)[:, 1].astype(np.float32)
        e_hat_te  = e_model.predict_proba(Xte)[:, 1].astype(np.float32)
        e_hat_res = np.clip(e_hat_res, 1e-6, 1 - 1e-6)
        e_hat_te  = np.clip(e_hat_te,  1e-6, 1 - 1e-6)

        m_model = make_nuisance_reg(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit)
        m_hat_res = m_model.predict(X_res).astype(np.float32)
        m_hat_te  = m_model.predict(Xte).astype(np.float32)

        y_target = (Y_res.astype(np.float32) - m_hat_res).astype(np.float32)
        ttil = (T_res.astype(np.float32) - e_hat_res).astype(np.float32)

        net = train_contY_net(X_res, T_res, y_target, ttil, seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)

        x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)

        # t=0
        t0 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
        tt0 = torch.from_numpy((0.0 - e_hat_te).astype(np.float32)).to(DEVICE)
        ytil0 = net(x_t, t0, tt0).cpu().numpy().astype(np.float32)
        mu0 = (m_hat_te + ytil0).astype(np.float32)

        # t=1
        t1 = torch.ones((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
        tt1 = torch.from_numpy((1.0 - e_hat_te).astype(np.float32)).to(DEVICE)
        ytil1 = net(x_t, t1, tt1).cpu().numpy().astype(np.float32)
        mu1 = (m_hat_te + ytil1).astype(np.float32)

        mu0s.append(mu0); mu1s.append(mu1)

    return np.mean(np.stack(mu0s, 0), 0), np.mean(np.stack(mu1s, 0), 0)

def orca_binT_binY_mu01(Xtr, Ttr, Ytr01, Xte, seed, nuisance_kind, tfeat_kind):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    mu0s, mu1s = [], []

    for fold_id, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
        X_fit, T_fit, Y_fit = Xtr[idx_fit], Ttr[idx_fit], Ytr01[idx_fit]
        X_res, T_res, Y_res = Xtr[idx_res], Ttr[idx_res], Ytr01[idx_res]

        e_model = make_nuisance_clf(nuisance_kind, seed + 20 + fold_id).fit(X_fit, T_fit.astype(int))
        e_hat_res = e_model.predict_proba(X_res)[:, 1].astype(np.float32)
        e_hat_te  = e_model.predict_proba(Xte)[:, 1].astype(np.float32)
        e_hat_res = np.clip(e_hat_res, 1e-6, 1 - 1e-6)
        e_hat_te  = np.clip(e_hat_te,  1e-6, 1 - 1e-6)

        m_model = make_nuisance_clf(nuisance_kind, seed + 10 + fold_id).fit(X_fit, Y_fit.astype(int))
        m_hat_res = m_model.predict_proba(X_res)[:, 1].astype(np.float32)
        m_hat_te  = m_model.predict_proba(Xte)[:, 1].astype(np.float32)

        base_logit_res = logit_np(m_hat_res)
        base_logit_te  = logit_np(m_hat_te)

        y_target = (Y_res.astype(np.float32) - m_hat_res).astype(np.float32)
        ttil = (T_res.astype(np.float32) - e_hat_res).astype(np.float32)

        # train delta-logit net with MSE on y_target? (keep same residual structure)
        # For stability, train contY net on y_target (as real-valued residual), then use base probability + residual,
        # but clip to [0,1]. This matches your previous binT_binY residual style.
        net = train_contY_net(X_res, T_res, y_target, ttil, seed=seed + 100 + fold_id, tfeat_kind=tfeat_kind)

        x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)

        t0 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
        tt0 = torch.from_numpy((0.0 - e_hat_te).astype(np.float32)).to(DEVICE)
        ytil0 = net(x_t, t0, tt0).cpu().numpy().astype(np.float32)
        mu0 = np.clip(m_hat_te + ytil0, 1e-6, 1 - 1e-6).astype(np.float32)

        t1 = torch.ones((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
        tt1 = torch.from_numpy((1.0 - e_hat_te).astype(np.float32)).to(DEVICE)
        ytil1 = net(x_t, t1, tt1).cpu().numpy().astype(np.float32)
        mu1 = np.clip(m_hat_te + ytil1, 1e-6, 1 - 1e-6).astype(np.float32)

        mu0s.append(mu0); mu1s.append(mu1)

    return np.mean(np.stack(mu0s, 0), 0), np.mean(np.stack(mu1s, 0), 0)

# NoOrth
def noorth_contT_contY_mu_grid(Xtr, Ttr, Ytr, Xte, seed, tfeat_kind):
    ttil = np.zeros_like(Ttr, dtype=np.float32)
    net = train_contY_net(Xtr, Ttr, Ytr.astype(np.float32), ttil, seed=seed + 777, tfeat_kind=tfeat_kind)
    x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)
    out = np.empty((Xte.shape[0], len(T_GRID)), dtype=np.float32)
    for j, tval in enumerate(T_GRID):
        t = np.full((Xte.shape[0],), tval, dtype=np.float32)
        tb = torch.from_numpy(t).to(DEVICE)
        tt = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
        out[:, j] = net(x_t, tb, tt).detach().cpu().numpy().astype(np.float32)
    return out

def noorth_contT_binY_p_grid(Xtr, Ttr, Ytr01, Xte, seed, tfeat_kind):
    net = train_binY_noorth_net(Xtr, Ttr, Ytr01.astype(np.float32), seed=seed + 888, tfeat_kind=tfeat_kind)
    x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)
    out = np.empty((Xte.shape[0], len(T_GRID)), dtype=np.float32)
    for j, tval in enumerate(T_GRID):
        t = np.full((Xte.shape[0],), tval, dtype=np.float32)
        tb = torch.from_numpy(t).to(DEVICE)
        tt = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
        logits = net(x_t, tb, tt).detach().cpu().numpy().astype(np.float32)
        out[:, j] = sigmoid_np(logits).astype(np.float32)
    return out

def noorth_binT_contY_mu01(Xtr, Ttr, Ytr, Xte, seed, tfeat_kind):
    ttil = np.zeros_like(Ttr, dtype=np.float32)
    net = train_contY_net(Xtr, Ttr, Ytr.astype(np.float32), ttil, seed=seed + 999, tfeat_kind=tfeat_kind)
    x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)

    t0 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    tt0 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    mu0 = net(x_t, t0, tt0).detach().cpu().numpy().astype(np.float32)

    t1 = torch.ones((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    tt1 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    mu1 = net(x_t, t1, tt1).detach().cpu().numpy().astype(np.float32)
    return mu0, mu1

def noorth_binT_binY_mu01(Xtr, Ttr, Ytr01, Xte, seed, tfeat_kind):
    net = train_binY_noorth_net(Xtr, Ttr, Ytr01.astype(np.float32), seed=seed + 1111, tfeat_kind=tfeat_kind)
    x_t = torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)

    t0 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    tt0 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    logit0 = net(x_t, t0, tt0).detach().cpu().numpy().astype(np.float32)
    mu0 = np.clip(sigmoid_np(logit0), 1e-6, 1 - 1e-6).astype(np.float32)

    t1 = torch.ones((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    tt1 = torch.zeros((Xte.shape[0],), dtype=torch.float32, device=DEVICE)
    logit1 = net(x_t, t1, tt1).detach().cpu().numpy().astype(np.float32)
    mu1 = np.clip(sigmoid_np(logit1), 1e-6, 1 - 1e-6).astype(np.float32)
    return mu0, mu1

# --------------------------
# 9) Resume helpers (per-task raw csv)
# --------------------------
RAW_COLUMNS = [
    "task","conf_strength","rep_id","train_seed","method","nuisance","tfeat","orth",
    "wall_time_sec",
    "m_ise","rmse_grid","sqrt_pehe","ate_err"
]

def load_done_keys(raw_csv):
    if not os.path.exists(raw_csv):
        return set(), []
    df = pd.read_csv(raw_csv, engine="python", on_bad_lines="skip")
    keys = set(zip(df.task, df.conf_strength, df.rep_id, df.train_seed, df.method))
    wall = df["wall_time_sec"].to_numpy(dtype=float) if "wall_time_sec" in df.columns else np.array([], dtype=float)
    return keys, wall.tolist()

def append_row(raw_csv, row: dict):
    # enforce fixed header order for robustness
    out = {k: row.get(k, np.nan) for k in RAW_COLUMNS}
    df1 = pd.DataFrame([out], columns=RAW_COLUMNS)
    if not os.path.exists(raw_csv):
        df1.to_csv(raw_csv, index=False)
    else:
        df1.to_csv(raw_csv, mode="a", header=False, index=False)

# --------------------------
# 10) Postprocess
# --------------------------
def make_heatmap(df_task, metric, out_dir, task, conf_strength):
    sub = df_task[(df_task["conf_strength"] == conf_strength) & (df_task["method"].str.startswith("ORCA["))].copy()
    if len(sub) == 0:
        return
    def parse_method(m):
        inside = m[len("ORCA["):-1]
        nuis, tpart = inside.split("|")
        tfeat = tpart.split("=")[1]
        return nuis.lower(), tfeat.lower()
    sub["nuis2"], sub["tfeat2"] = zip(*sub["method"].map(parse_method))

    mat = sub.groupby(["nuis2","tfeat2"])[metric].mean().reset_index()
    nuis_order = ["rf","ridge","mlp","gbdt"]
    tfeat_order = ["fourier","direct","mlp"]
    piv = mat.pivot(index="nuis2", columns="tfeat2", values=metric).reindex(index=nuis_order, columns=tfeat_order)

    plt.figure(figsize=(6.2, 4.2))
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

def postprocess_task(task_dir, task):
    RAW = os.path.join(task_dir, f"raw_{task}.csv")
    if not os.path.exists(RAW):
        return
    df = pd.read_csv(RAW, engine="python", on_bad_lines="skip")
    if len(df) == 0:
        return

    # pick primary metric
    if task.startswith("contT_"):
        primary = "m_ise"
        caption_metric = "MISE"
    else:
        primary = "sqrt_pehe"
        caption_metric = "\\\\sqrt{\\\\epsilon_{\\\\mathrm{PEHE}}}"

    # summary
    SUM = os.path.join(task_dir, f"summary_{task}.csv")
    summ = df.groupby(["conf_strength","method"])[["m_ise","rmse_grid","sqrt_pehe","ate_err"]].agg(["mean","std","count"]).reset_index()
    summ.to_csv(SUM, index=False)
    print("[Post]", task, "saved:", SUM)

    # topk latex
    TOPK_TEX = os.path.join(task_dir, f"table_topk_{task}.tex")
    tmp = summ.copy()
    tmp.columns = [
        "conf_strength","method",
        "m_ise_mean","m_ise_std","m_ise_count",
        "rmse_grid_mean","rmse_grid_std","rmse_grid_count",
        "sqrt_pehe_mean","sqrt_pehe_std","sqrt_pehe_count",
        "ate_err_mean","ate_err_std","ate_err_count",
    ]

    TOPK = 8
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(f"conf & Method & {caption_metric} (mean$\\pm$sd) & $n$ \\\\")
    lines.append("\\midrule")

    for cs in sorted(tmp.conf_strength.unique()):
        sub = tmp[tmp.conf_strength == cs].copy()
        sub = sub.sort_values(f"{primary}_mean").head(TOPK)
        first = True
        for _, r in sub.iterrows():
            conf_cell = f"{cs:g}" if first else ""
            sd = 0.0 if np.isnan(r[f"{primary}_std"]) else float(r[f"{primary}_std"])
            cell = f"{float(r[f'{primary}_mean']):.4g}$\\pm${sd:.3g}"
            lines.append(f"{conf_cell} & {latex_escape(r['method'])} & {cell} & {int(r[f'{primary}_count'])} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{NEWS ablations ({task}). Lower is better.}}")
    lines.append(f"\\label{{tab:news_{task}_topk}}")
    lines.append("\\end{table}")
    with open(TOPK_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[Post]", task, "wrote:", TOPK_TEX)

    # heatmaps per conf (ORCA only)
    for cs in sorted(df["conf_strength"].unique()):
        make_heatmap(df, primary, task_dir, task, cs)

    # paired bootstrap CI: best ORCA per conf vs all other methods (including NoOrth)
    CI = os.path.join(task_dir, f"paired_bootstrap_ci_{task}.csv")
    orc_mean = (df[df["method"].str.startswith("ORCA[")]
                  .groupby(["conf_strength","method"])[primary]
                  .mean()
                  .reset_index())
    if len(orc_mean) == 0:
        return
    best_orca = (orc_mean.sort_values(["conf_strength", primary])
                        .groupby("conf_strength")
                        .head(1)
                        .rename(columns={"method":"best_method"}))

    rows = []
    for cs in sorted(df["conf_strength"].unique()):
        best = best_orca[best_orca.conf_strength == cs].iloc[0]["best_method"]
        sub = df[df.conf_strength == cs].copy()
        methods = sorted(sub["method"].unique())
        for b in methods:
            if b == best:
                continue
            res = paired_delta(sub, best, b, metric=primary, n_boot=4000, seed=int(float(cs)*1000)+19)
            if res is None:
                continue
            rows.append({
                "conf_strength": float(cs),
                "best_orca": best,
                "baseline": b,
                "metric": primary,
                "comparison": f"{best} - {b}",
                **res
            })
    pd.DataFrame(rows).to_csv(CI, index=False)
    print("[Post]", task, "saved:", CI)

# --------------------------
# 11) Main
# --------------------------
def main():
    # load data
    print("Loading News BOW...")
    X_base = load_news_bow(n_docs=N_DOCS, top_vocab=TOP_VOCAB, cache_dir=DATA_CACHE_DIR)
    print("X_base:", X_base.shape)

    for task in TASKS:
        task_dir = os.path.join(OUT_DIR, f"TASK_{task}")
        os.makedirs(task_dir, exist_ok=True)

        RAW = os.path.join(task_dir, f"raw_{task}.csv")
        done, wall_hist = load_done_keys(RAW)

        # planned rows
        planned_orca = len(CONF_LIST) * len(REP_IDS) * len(TRAIN_SEEDS) * len(NUIS_KINDS) * len(TFEAT_KINDS)
        planned_noorth = len(CONF_LIST) * len(REP_IDS) * len(TRAIN_SEEDS) * len(NOORTH_TFEATS)
        planned = planned_orca + planned_noorth

        print(f"\n========== TASK={task} | resume rows={len(done)} | planned={planned} ==========")

        t0 = time.time()
        last_print = time.time()

        for cs in CONF_LIST:
            for rep_id in REP_IDS:
                if task.startswith("contT_"):
                    Xtr, Ttr, Ytr, Xte, Tte, Yte, mu_grid_te = get_split_cached_news(task, X_base, rep_id, cs)
                else:
                    Xtr, Ttr, Ytr, Xte, Tte, Yte, mu0_te, mu1_te = get_split_cached_news(task, X_base, rep_id, cs)

                for seed in TRAIN_SEEDS:
                    # ----------------- ORCA 12 -----------------
                    for nuis in NUIS_KINDS:
                        for tfeat in TFEAT_KINDS:
                            method = f"ORCA[{nuis.upper()}|t={tfeat}]"
                            key = (task, float(cs), int(rep_id), int(seed), method)
                            if key in done:
                                continue
                            start = time.time()

                            row = {
                                "task": task,
                                "conf_strength": float(cs),
                                "rep_id": int(rep_id),
                                "train_seed": int(seed),
                                "method": method,
                                "nuisance": nuis,
                                "tfeat": tfeat,
                                "orth": True,
                            }

                            # compute metrics
                            if task == "contT_contY":
                                mu_hat = orca_contT_contY_mu_grid(Xtr, Ttr, Ytr, Xte, seed=seed, nuisance_kind=nuis, tfeat_kind=tfeat)
                                row["m_ise"] = mise_grid(mu_hat, mu_grid_te)
                                row["rmse_grid"] = rmse(mu_hat, mu_grid_te)
                                row["sqrt_pehe"] = np.nan
                                row["ate_err"] = np.nan

                            elif task == "contT_binY":
                                p_hat = orca_contT_binY_p_grid(Xtr, Ttr, Ytr, Xte, seed=seed, nuisance_kind=nuis, tfeat_kind=tfeat)
                                row["m_ise"] = mise_grid(p_hat, mu_grid_te)
                                row["rmse_grid"] = rmse(p_hat, mu_grid_te)
                                row["sqrt_pehe"] = np.nan
                                row["ate_err"] = np.nan

                            elif task == "binT_contY":
                                mu0_hat, mu1_hat = orca_binT_contY_mu01(Xtr, Ttr, Ytr, Xte, seed=seed, nuisance_kind=nuis, tfeat_kind=tfeat)
                                ite_hat = (mu1_hat - mu0_hat).astype(np.float32)
                                ite_true = (mu1_te - mu0_te).astype(np.float32)
                                row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                                row["ate_err"] = float(abs(np.mean(ite_hat) - np.mean(ite_true)))
                                row["m_ise"] = np.nan
                                row["rmse_grid"] = np.nan

                            elif task == "binT_binY":
                                mu0_hat, mu1_hat = orca_binT_binY_mu01(Xtr, Ttr, Ytr, Xte, seed=seed, nuisance_kind=nuis, tfeat_kind=tfeat)
                                ite_hat = (mu1_hat - mu0_hat).astype(np.float32)
                                ite_true = (mu1_te - mu0_te).astype(np.float32)
                                row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                                row["ate_err"] = float(abs(np.mean(ite_hat) - np.mean(ite_true)))
                                row["m_ise"] = np.nan
                                row["rmse_grid"] = np.nan

                            else:
                                raise ValueError(task)

                            row["wall_time_sec"] = float(time.time() - start)
                            append_row(RAW, row)
                            done.add(key)
                            wall_hist.append(row["wall_time_sec"])

                            if time.time() - last_print > 60:
                                total_done = len(done)
                                elapsed = time.time() - t0
                                if len(wall_hist) >= 10:
                                    med = float(np.median(wall_hist[-min(300, len(wall_hist)):]))
                                    remaining = planned - total_done
                                    eta_sec = remaining * med
                                    print(f"[Progress] {total_done}/{planned} | elapsed={elapsed/3600:.2f}h | median_row={med:.2f}s | ETA~{eta_sec/3600:.2f}h")
                                else:
                                    print(f"[Progress] {total_done}/{planned} | elapsed={elapsed/3600:.2f}h")
                                last_print = time.time()

                    # ----------------- NoOrth 3 -----------------
                    for tfeat in NOORTH_TFEATS:
                        method = f"NoOrth[t={tfeat}]"
                        key = (task, float(cs), int(rep_id), int(seed), method)
                        if key in done:
                            continue
                        start = time.time()

                        row = {
                            "task": task,
                            "conf_strength": float(cs),
                            "rep_id": int(rep_id),
                            "train_seed": int(seed),
                            "method": method,
                            "nuisance": "na",
                            "tfeat": tfeat,
                            "orth": False,
                        }

                        if task == "contT_contY":
                            mu_hat = noorth_contT_contY_mu_grid(Xtr, Ttr, Ytr, Xte, seed=seed, tfeat_kind=tfeat)
                            row["m_ise"] = mise_grid(mu_hat, mu_grid_te)
                            row["rmse_grid"] = rmse(mu_hat, mu_grid_te)
                            row["sqrt_pehe"] = np.nan
                            row["ate_err"] = np.nan

                        elif task == "contT_binY":
                            p_hat = noorth_contT_binY_p_grid(Xtr, Ttr, Ytr, Xte, seed=seed, tfeat_kind=tfeat)
                            row["m_ise"] = mise_grid(p_hat, mu_grid_te)
                            row["rmse_grid"] = rmse(p_hat, mu_grid_te)
                            row["sqrt_pehe"] = np.nan
                            row["ate_err"] = np.nan

                        elif task == "binT_contY":
                            mu0_hat, mu1_hat = noorth_binT_contY_mu01(Xtr, Ttr, Ytr, Xte, seed=seed, tfeat_kind=tfeat)
                            ite_hat = (mu1_hat - mu0_hat).astype(np.float32)
                            ite_true = (mu1_te - mu0_te).astype(np.float32)
                            row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                            row["ate_err"] = float(abs(np.mean(ite_hat) - np.mean(ite_true)))
                            row["m_ise"] = np.nan
                            row["rmse_grid"] = np.nan

                        elif task == "binT_binY":
                            mu0_hat, mu1_hat = noorth_binT_binY_mu01(Xtr, Ttr, Ytr, Xte, seed=seed, tfeat_kind=tfeat)
                            ite_hat = (mu1_hat - mu0_hat).astype(np.float32)
                            ite_true = (mu1_te - mu0_te).astype(np.float32)
                            row["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
                            row["ate_err"] = float(abs(np.mean(ite_hat) - np.mean(ite_true)))
                            row["m_ise"] = np.nan
                            row["rmse_grid"] = np.nan

                        else:
                            raise ValueError(task)

                        row["wall_time_sec"] = float(time.time() - start)
                        append_row(RAW, row)
                        done.add(key)
                        wall_hist.append(row["wall_time_sec"])

        print(f"Done TASK={task}. RAW={RAW}")
        postprocess_task(task_dir, task)

    print("\nAll outputs saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
