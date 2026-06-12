# -*- coding: utf-8 -*-
"""
ICML 2020 (Schwab et al.) ICU benchmark-style synthetic DGP,
adapted to YOUR Chloride dataset.

v3 changes (for better continuous performance):
1) ORCA now uses the standard orthogonal form:
   mu(x,s) = m(x) + (s - e(x)) * tau(x,s)
   and trains tau-net with loss on ytil ~= (s-e)*tau.
2) Add ORCA-Stack meta learner (continuous only, default ON in --fast):
   meta learns y ~ f([X, s, mu_orca]) using OOF ORCA factual preds.
3) CSV writer uses fixed columns -> no more pandas ParserError.
4) Optional y_transform for *_duration: none|log1p (apply to y and true curve consistently).

+ v3.1 additions (this file):
5) Add DRNet and VCNet baselines (torch) that output full dose-response curve:
   - DRNet: shared repr + multiple heads by dosage bins, with linear interpolation.
   - VCNet: varying-coefficient net using cubic B-spline basis in s.

Evaluation:
- continuous: MISE over s_grid (curve), RMSE over grid
- binary: MISE over prob curve + factual AUC/logloss at observed s
"""

import os, math, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss

# -------------------------
# Optional deps
# -------------------------
HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HAS_BART = False
try:
    from bartpy.sklearnmodel import SklearnModel as BartModel
    HAS_BART = True
except Exception:
    HAS_BART = False


# -------------------------
# Utils
# -------------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def gaussian(x, mean, std):
    std = max(float(std), 1e-6)
    x = float(x)
    mean = float(mean)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))

def stable_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = np.sum(exps)
    if s <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / s

def mise_curve(pred, true, grid):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    dt = float(grid[1] - grid[0])
    return float(np.mean(np.sum((pred - true) ** 2, axis=1) * dt))

def rmse_grid(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def winsorize_series(s: pd.Series, q: float):
    lo = s.quantile(q)
    hi = s.quantile(1.0 - q)
    return s.clip(lo, hi)

def quantile_map(src_values, ref_values):
    src = np.asarray(src_values, dtype=np.float64)
    ref = np.asarray(ref_values, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.size < 10:
        return src.copy()
    qs = np.clip(src, 0.0, 1.0)
    return np.quantile(ref, qs).astype(np.float64)

def monotone_match_distribution(y_raw, y_target, n_q=2048):
    y_raw = np.asarray(y_raw, dtype=np.float64)
    y_tar = np.asarray(y_target, dtype=np.float64)
    y_tar = y_tar[np.isfinite(y_tar)]
    if y_tar.size < 50:
        return y_raw.astype(np.float32)

    qs = np.linspace(0.0, 1.0, n_q)
    q_raw = np.quantile(y_raw, qs)
    q_tar = np.quantile(y_tar, qs)
    mapped = np.interp(y_raw, q_raw, q_tar, left=q_tar[0], right=q_tar[-1])
    return mapped.astype(np.float32)

def solve_bias_for_prevalence(logits, target_prev, iters=60):
    target_prev = float(np.clip(target_prev, 1e-6, 1 - 1e-6))
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = float(np.mean(sigmoid(logits + mid)))
        if p < target_prev:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def maybe_transform_duration(y, kind: str):
    """
    Apply transform for continuous duration tasks.
    IMPORTANT: must apply consistently to y_tr/y_te and po_te.
    """
    kind = (kind or "none").lower()
    y = np.asarray(y, dtype=np.float64)
    if kind == "none":
        return y.astype(np.float32)
    if kind == "log1p":
        y2 = np.clip(y, 0.0, None)
        return np.log1p(y2).astype(np.float32)
    raise ValueError(f"Unknown y_transform: {kind}")


# -------------------------
# Load + clean YOUR dataset
# -------------------------
def load_and_clean(data_path: str,
                   t_col: str,
                   y_cols: list,
                   dropna_rows: bool = True,
                   winsor_q: float = 0.01,
                   max_n: int | None = None,
                   seed: int = 123):
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in [t_col] + y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    drop_cols = set([t_col] + y_cols)
    cov_cols = [c for c in df.columns if c not in drop_cols]

    # Convert non-numeric covariates to category codes
    for c in cov_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)

    for c in [t_col] + y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if dropna_rows:
        df = df.dropna(subset=[t_col] + y_cols)

    for c in cov_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    if winsor_q is not None and winsor_q > 0:
        for c in cov_cols + [t_col]:
            df[c] = winsorize_series(df[c], winsor_q)

    if max_n is not None and len(df) > int(max_n):
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(len(df)), size=int(max_n), replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    return df.reset_index(drop=True), cov_cols


# -------------------------
# ICU2020 benchmark-style DGP (exposure only)
# -------------------------
class ICU2020_DGP_ExposureOnly:
    def __init__(self,
                 X_pool_std: np.ndarray,
                 seed_fit: int = 909,
                 response_mean_of_mean: float = 0.45,
                 response_std_of_mean: float = 0.15,
                 response_mean_of_std: float = 0.10,
                 response_std_of_std: float = 0.05,
                 kappa: float = 10.0,
                 epsilon_std: float = 0.15,
                 scaling_constant: float = 150.0,
                 scaling_offset: float = 50.0,
                 treatment_mean: float = 0.60):
        self.X_pool = np.asarray(X_pool_std, dtype=np.float32)
        self.rs = np.random.RandomState(int(seed_fit))

        self.response_mean_of_mean = float(response_mean_of_mean)
        self.response_std_of_mean = float(response_std_of_mean)
        self.response_mean_of_std = float(response_mean_of_std)
        self.response_std_of_std = float(response_std_of_std)

        self.kappa = float(kappa)
        self.epsilon_std = float(epsilon_std)
        self.C = float(scaling_constant)
        self.offset = float(scaling_offset)

        self.num_archetypes = 2
        self.treatment_mean = float(treatment_mean)

        self.centroids = None            # length 2: treat + control
        self.dosage_centroids = None     # 2 archetypes for treat

    def _sample_resp_params(self, adjust_last: bool):
        mean_of_mean = (1.0 - self.response_mean_of_mean) if adjust_last else self.response_mean_of_mean
        r_mean = clip01(self.rs.normal(mean_of_mean, self.response_std_of_mean))
        r_std = clip01(self.rs.normal(self.response_mean_of_std, self.response_std_of_std)) + 0.025
        return float(r_mean), float(r_std)

    def fit(self):
        n = self.X_pool.shape[0]
        idx = self.rs.permutation(n)[:2]
        idx = list(map(int, idx))

        c0 = self.X_pool[idx[0]].astype(np.float32)
        r0_mean, r0_std = self._sample_resp_params(adjust_last=False)

        c1 = self.X_pool[idx[1]].astype(np.float32)
        r1_mean, r1_std = self._sample_resp_params(adjust_last=True)

        self.centroids = [(c0, r0_mean, r0_std), (c1, r1_mean, r1_std)]

        arch_idx = self.rs.permutation(n)[:self.num_archetypes]
        arch_idx = list(map(int, arch_idx))
        arches = []
        for aidx in arch_idx:
            cvec = self.X_pool[aidx].astype(np.float32)
            d_mean, d_std = self._sample_resp_params(adjust_last=False)
            d_min = float(self.rs.normal(0.0, 0.1))
            arches.append((cvec, d_mean, d_std, d_min))
        self.dosage_centroids = arches

    @staticmethod
    def _euclid_dist(x: np.ndarray, c: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        return float(np.sqrt(np.sum((x - c) ** 2)))

    def get_centroid_weights(self, x: np.ndarray, centroids_vecs: list):
        dists = [self._euclid_dist(x, c) for c in centroids_vecs]
        return np.asarray(dists, dtype=np.float64)

    def expected_responses(self):
        exp = []
        for j in range(2):
            _, r_mean, r_std = self.centroids[j]
            y_this = float(self.rs.normal(r_mean, r_std))
            exp.append(clip01(y_this + float(self.rs.normal(0.0, self.epsilon_std))))
        return np.asarray(exp, dtype=np.float64)

    def dose_response_curve_params(self, x: np.ndarray):
        arches = self.dosage_centroids
        dists = self.get_centroid_weights(x, [arches[0][0], arches[1][0]])
        d = stable_softmax(self.kappa * dists)  # keep same sign as snippet
        _, d0_mean, d0_std, d0_min = arches[0]
        _, d1_mean, d1_std, d1_min = arches[1]
        return d, d0_mean, d0_std, d0_min, d1_mean, d1_std, d1_min

    def dose_response_value(self, s: float, params):
        d, d0_mean, d0_std, d0_min, d1_mean, d1_std, d1_min = params
        v = float(d[0]) * gaussian(float(s) - float(d0_min), float(d0_mean), float(d0_std)) + \
            float(d[1]) * gaussian(float(s) - float(d1_min), float(d1_mean), float(d1_std))
        return float(v)

    def simulate_curves(self, X_std: np.ndarray, s_grid: np.ndarray):
        assert self.centroids is not None and self.dosage_centroids is not None

        X_std = np.asarray(X_std, dtype=np.float32)
        s_grid = np.asarray(s_grid, dtype=np.float32)
        n = X_std.shape[0]

        s_obs = np.zeros(n, dtype=np.float32)
        y_obs = np.zeros(n, dtype=np.float32)
        po = np.zeros((n, len(s_grid)), dtype=np.float32)

        for i in range(n):
            x = X_std[i]
            exp = self.expected_responses()
            exp_treat = float(exp[0])

            s = clip01(float(self.rs.normal(self.treatment_mean, 0.1)))
            s_obs[i] = float(s)

            params = self.dose_response_curve_params(x)

            y = self.dose_response_value(s, params) * exp_treat
            y_obs[i] = float(self.offset + self.C * y)

            for j, sj in enumerate(s_grid):
                yj = self.dose_response_value(float(sj), params) * exp_treat
                po[i, j] = float(self.offset + self.C * yj)

        return s_obs, y_obs, po


# -------------------------
# ORCA tau-net (standard form)
# -------------------------
class TauNet(nn.Module):
    def __init__(self, x_dim: int, hidden: int, rep_dim: int, dropout: float,
                 t_mode: str = "direct", n_freq: int = 10, tmlp_dim: int = 32):
        super().__init__()
        self.t_mode = str(t_mode).lower()
        self.n_freq = int(n_freq)

        self.x_net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep_dim), nn.ReLU(),
        )

        if self.t_mode == "direct":
            self.t_net = None
            t_rep_dim = 1
        elif self.t_mode == "fourier":
            t_rep_dim = 2 * self.n_freq
            self.t_net = None
        elif self.t_mode == "mlp":
            self.t_net = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, tmlp_dim), nn.ReLU(),
            )
            t_rep_dim = int(tmlp_dim)
        else:
            raise ValueError(f"Unknown t_mode: {t_mode}")

        self.head = nn.Sequential(
            nn.Linear(rep_dim + t_rep_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _t_features(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,1) with s in [0,1]
        if self.t_mode == "direct":
            return t
        if self.t_mode == "mlp":
            return self.t_net(t)
        # fourier
        k = torch.arange(1, self.n_freq + 1, device=t.device, dtype=t.dtype).view(1, -1)
        ang = 2.0 * math.pi * t * k
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return feats

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        xr = self.x_net(x)
        tf = self._t_features(s)
        h = torch.cat([xr, tf], dim=1)
        return self.head(h)  # tau(x,s)


def make_nuisance(kind: str, seed: int, task: str):
    k = kind.lower()
    rs = int(seed)

    if k == "ridge":
        if task == "reg":
            return Ridge(alpha=1.0)
        else:
            return LogisticRegression(max_iter=300, n_jobs=1)

    if k == "rf":
        if task == "reg":
            return RandomForestRegressor(n_estimators=250, random_state=rs, n_jobs=-1, min_samples_leaf=10)
        else:
            return RandomForestClassifier(n_estimators=250, random_state=rs, n_jobs=-1, min_samples_leaf=10)

    if k == "gbdt":
        if task == "reg":
            return HistGradientBoostingRegressor(random_state=rs, max_iter=350, learning_rate=0.05, max_depth=6)
        else:
            return HistGradientBoostingClassifier(random_state=rs, max_iter=350, learning_rate=0.05, max_depth=6)

    if k == "mlp":
        if task == "reg":
            return MLPRegressor(hidden_layer_sizes=(256, 256), random_state=rs, max_iter=250, early_stopping=True)
        else:
            return MLPClassifier(hidden_layer_sizes=(256, 256), random_state=rs, max_iter=250, early_stopping=True)

    if k == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available.")
        if task == "reg":
            return xgb.XGBRegressor(
                n_estimators=900, max_depth=7, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=rs,
                tree_method="hist"
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=900, max_depth=7, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=rs,
                tree_method="hist",
                eval_metric="logloss"
            )

    raise ValueError(f"Unknown nuisance kind: {kind}")


def train_tau_net(X, s, ytil, ttil, seed, device,
                  t_mode="direct",
                  epochs=60, batch=2048, lr=1e-3, wd=1e-5,
                  hidden=256, rep_dim=128, dropout=0.1, n_freq=10, tmlp_dim=32):
    """
    Train tau-net by minimizing MSE( ttil * tau(x,s), ytil ).
    """
    set_all_seeds(seed)

    X = np.asarray(X, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32).reshape(-1, 1)
    ytil = np.asarray(ytil, dtype=np.float32).reshape(-1, 1)
    ttil = np.asarray(ttil, dtype=np.float32).reshape(-1, 1)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(s), torch.from_numpy(ytil), torch.from_numpy(ttil))
    dl = DataLoader(ds, batch_size=min(int(batch), len(X)), shuffle=True, drop_last=False)

    net = TauNet(
        x_dim=X.shape[1], hidden=int(hidden), rep_dim=int(rep_dim),
        dropout=float(dropout), t_mode=str(t_mode), n_freq=int(n_freq), tmlp_dim=int(tmlp_dim)
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=float(wd))
    mse = nn.MSELoss()

    net.train()
    for _ in range(int(epochs)):
        for xb, sb, yb, ttb in dl:
            xb = xb.to(device)
            sb = sb.to(device)
            yb = yb.to(device)
            ttb = ttb.to(device)
            opt.zero_grad(set_to_none=True)
            tau = net(xb, sb)
            pred_ytil = tau * ttb
            loss = mse(pred_ytil, yb)
            loss.backward()
            opt.step()

    net.eval()
    return net


@torch.no_grad()
def predict_mu_grid(net, m_model, e_model, X, s_grid, device, is_binary: bool):
    X = np.asarray(X, dtype=np.float32)
    s_grid = np.asarray(s_grid, dtype=np.float32).reshape(-1)

    if is_binary:
        m = m_model.predict_proba(X)[:, 1].astype(np.float32)
    else:
        m = m_model.predict(X).astype(np.float32)

    e = e_model.predict(X).astype(np.float32)

    xb = torch.from_numpy(X).to(device)

    outs = []
    for sv in s_grid:
        sb = torch.full((X.shape[0], 1), float(sv), dtype=torch.float32, device=device)
        tau = net(xb, sb).squeeze(1).detach().cpu().numpy().astype(np.float32)
        mu = m + (float(sv) - e) * tau
        if is_binary:
            mu = np.clip(mu, 0.0, 1.0)
        outs.append(mu[:, None])
    return np.concatenate(outs, axis=1).astype(np.float32)


def orca_fit_predict_curve(Xtr, s_tr, y_tr, Xte, s_grid,
                           seed: int,
                           nuis: str,
                           t_mode: str,
                           device,
                           n_folds: int = 2,
                           epochs: int = 60,
                           fast: bool = False,
                           is_binary: bool = False,
                           return_oof_factual: bool = False):
    """
    Proper cross-fitting:
      For each fold:
        - fit m,e on idx_fit
        - train tau-net on idx_fit residuals
        - predict curve on Xte (average over folds)
        - produce OOF factual mu_hat for idx_res (optional)
    """
    set_all_seeds(seed)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    s_tr = np.asarray(s_tr, dtype=np.float32).reshape(-1)
    y_tr = np.asarray(y_tr, dtype=np.float32).reshape(-1)
    Xte = np.asarray(Xte, dtype=np.float32)

    kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))

    pred_curves = []
    oof_mu_factual = np.full(len(Xtr), np.nan, dtype=np.float32)

    for fold, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
        X_fit, s_fit, y_fit = Xtr[idx_fit], s_tr[idx_fit], y_tr[idx_fit]
        X_res, s_res, y_res = Xtr[idx_res], s_tr[idx_res], y_tr[idx_res]

        # m(x)
        if is_binary:
            m_model = make_nuisance(nuis, seed + 1000 + fold, task="clf")
            m_model.fit(X_fit, y_fit)
            m_fit = m_model.predict_proba(X_fit)[:, 1].astype(np.float32)
            m_res = m_model.predict_proba(X_res)[:, 1].astype(np.float32)
        else:
            m_model = make_nuisance(nuis, seed + 1000 + fold, task="reg")
            m_model.fit(X_fit, y_fit)
            m_fit = m_model.predict(X_fit).astype(np.float32)
            m_res = m_model.predict(X_res).astype(np.float32)

        # e(x)
        e_model = make_nuisance(nuis, seed + 2000 + fold, task="reg")
        e_model.fit(X_fit, s_fit)
        e_fit = e_model.predict(X_fit).astype(np.float32)
        e_res = e_model.predict(X_res).astype(np.float32)

        ytil_fit = (y_fit - m_fit).astype(np.float32)
        ttil_fit = (s_fit - e_fit).astype(np.float32)

        ep = int(epochs)
        if fast:
            ep = min(ep, 30)

        net = train_tau_net(
            X_fit, s_fit, ytil_fit, ttil_fit,
            seed=seed + 5000 + fold,
            device=device,
            t_mode=t_mode,
            epochs=ep,
            batch=2048 if fast else 4096,
            lr=1e-3, wd=1e-5,
            hidden=256, rep_dim=128, dropout=0.1,
            n_freq=10, tmlp_dim=32
        )

        # OOF factual mu on idx_res
        if return_oof_factual:
            with torch.no_grad():
                xb = torch.from_numpy(X_res).to(device)
                sb = torch.from_numpy(s_res.reshape(-1, 1)).to(device)
                tau_res = net(xb, sb).squeeze(1).detach().cpu().numpy().astype(np.float32)
            mu_res = m_res + (s_res - e_res) * tau_res
            if is_binary:
                mu_res = np.clip(mu_res, 0.0, 1.0)
            oof_mu_factual[idx_res] = mu_res

        # curve prediction on Xte
        pred_curve = predict_mu_grid(net, m_model, e_model, Xte, s_grid, device, is_binary=is_binary)
        pred_curves.append(pred_curve)

    pred_curve_mean = np.mean(pred_curves, axis=0).astype(np.float32)

    if return_oof_factual:
        return pred_curve_mean, oof_mu_factual
    return pred_curve_mean


# -------------------------
# Baselines
# -------------------------
def baseline_direct_regression(Xtr, s_tr, y_tr, Xte, s_grid, model_kind="xgb", is_binary=False, seed=0):
    Ztr = np.concatenate([Xtr, s_tr.reshape(-1, 1)], axis=1).astype(np.float32)
    preds = []

    if is_binary:
        if model_kind == "xgb":
            if not HAS_XGB:
                raise RuntimeError("xgboost not available.")
            model = xgb.XGBClassifier(
                n_estimators=900, max_depth=7, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=int(seed),
                tree_method="hist",
                eval_metric="logloss"
            )
        elif model_kind == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(256, 256), random_state=int(seed), max_iter=250, early_stopping=True)
        else:
            model = LogisticRegression(max_iter=300)
        model.fit(Ztr, y_tr)

        for sv in s_grid:
            Zte = np.concatenate([Xte, np.full((Xte.shape[0], 1), float(sv), dtype=np.float32)], axis=1)
            p = model.predict_proba(Zte)[:, 1].astype(np.float32)
            preds.append(p[:, None])
        return np.concatenate(preds, axis=1).astype(np.float32)

    else:
        if model_kind == "xgb":
            if not HAS_XGB:
                raise RuntimeError("xgboost not available.")
            model = xgb.XGBRegressor(
                n_estimators=1200, max_depth=7, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=int(seed),
                tree_method="hist"
            )
        elif model_kind == "rf":
            model = RandomForestRegressor(n_estimators=350, random_state=int(seed), n_jobs=-1, min_samples_leaf=10)
        elif model_kind == "ridge":
            model = Ridge(alpha=1.0)
        else:
            model = HistGradientBoostingRegressor(random_state=int(seed), max_iter=450, learning_rate=0.05, max_depth=6)

        model.fit(Ztr, y_tr)

        for sv in s_grid:
            Zte = np.concatenate([Xte, np.full((Xte.shape[0], 1), float(sv), dtype=np.float32)], axis=1)
            mu = model.predict(Zte).astype(np.float32)
            preds.append(mu[:, None])
        return np.concatenate(preds, axis=1).astype(np.float32)


def baseline_mlp_on_xt(Xtr, s_tr, y_tr, Xte, s_grid, is_binary=False, seed=0):
    return baseline_direct_regression(Xtr, s_tr, y_tr, Xte, s_grid, model_kind="mlp", is_binary=is_binary, seed=seed)


def baseline_gps(Xtr, s_tr, y_tr, Xte, s_grid, is_binary=False, seed=0):
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    s_tr = np.asarray(s_tr, np.float32).reshape(-1)
    y_tr = np.asarray(y_tr, np.float32).reshape(-1)

    # e(x)
    if HAS_XGB:
        ex = xgb.XGBRegressor(
            n_estimators=700, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=int(seed),
            tree_method="hist"
        )
    else:
        ex = HistGradientBoostingRegressor(random_state=int(seed), max_iter=350, learning_rate=0.05, max_depth=6)

    ex.fit(Xtr, s_tr)
    ehat_tr = ex.predict(Xtr).astype(np.float64)
    resid = s_tr.astype(np.float64) - ehat_tr
    sigma = float(np.std(resid) + 1e-6)

    def gps_density(s, mean):
        return gaussian(float(s), float(mean), float(sigma))

    gps_tr = np.array([gps_density(s_tr[i], ehat_tr[i]) for i in range(len(s_tr))], dtype=np.float64)

    F_tr = np.stack([s_tr, s_tr**2, gps_tr, gps_tr**2, s_tr * gps_tr], axis=1).astype(np.float64)

    if is_binary:
        out = LogisticRegression(max_iter=400)
        out.fit(F_tr, y_tr.astype(int))
    else:
        out = Ridge(alpha=1.0)
        out.fit(F_tr, y_tr.astype(np.float64))

    ehat_te = ex.predict(Xte).astype(np.float64)
    preds = []
    for sv in s_grid:
        sv = float(sv)
        gps_te = np.array([gps_density(sv, ehat_te[i]) for i in range(len(ehat_te))], dtype=np.float64)
        F_te = np.stack([
            np.full_like(gps_te, sv),
            np.full_like(gps_te, sv*sv),
            gps_te,
            gps_te**2,
            sv * gps_te
        ], axis=1).astype(np.float64)
        if is_binary:
            p = out.predict_proba(F_te)[:, 1].astype(np.float32)
            preds.append(p[:, None])
        else:
            mu = out.predict(F_te).astype(np.float32)
            preds.append(mu[:, None])
    return np.concatenate(preds, axis=1).astype(np.float32)


def baseline_bart(Xtr, s_tr, y_tr, Xte, s_grid, is_binary=False, seed=0):
    if not HAS_BART:
        raise RuntimeError("bartpy not available.")
    Ztr = np.concatenate([Xtr, s_tr.reshape(-1, 1)], axis=1).astype(np.float64)
    model = BartModel()
    model.fit(Ztr, y_tr.astype(np.float64))

    preds = []
    for sv in s_grid:
        Zte = np.concatenate([Xte, np.full((Xte.shape[0], 1), float(sv), dtype=np.float32)], axis=1).astype(np.float64)
        mu = model.predict(Zte).astype(np.float32)
        if is_binary:
            mu = np.clip(mu, 0.0, 1.0)
        preds.append(mu[:, None])
    return np.concatenate(preds, axis=1).astype(np.float32)


# -------------------------
# ORCA-Stack meta learner (continuous only)
# -------------------------
def fit_stack_meta(Xtr, s_tr, mu_orca_oof, y_tr, seed=0, kind="xgb"):
    Z = np.concatenate([Xtr, s_tr.reshape(-1, 1), mu_orca_oof.reshape(-1, 1)], axis=1).astype(np.float32)
    kind = (kind or "xgb").lower()

    if kind == "xgb":
        if not HAS_XGB:
            kind = "gbdt"
        else:
            model = xgb.XGBRegressor(
                n_estimators=1400, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=int(seed),
                tree_method="hist"
            )
            model.fit(Z, y_tr.astype(np.float32))
            return model, kind

    if kind == "gbdt":
        model = HistGradientBoostingRegressor(random_state=int(seed), max_iter=600, learning_rate=0.03, max_depth=6)
        model.fit(Z, y_tr.astype(np.float32))
        return model, kind

    if kind == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(Z, y_tr.astype(np.float32))
        return model, kind

    raise ValueError(f"Unknown stack kind: {kind}")

def predict_stack_curve(meta_model, Xte, s_grid, mu_orca_curve):
    preds = []
    for j, sv in enumerate(s_grid):
        Zte = np.concatenate([
            Xte.astype(np.float32),
            np.full((Xte.shape[0], 1), float(sv), dtype=np.float32),
            mu_orca_curve[:, j].reshape(-1, 1).astype(np.float32),
        ], axis=1)
        yhat = meta_model.predict(Zte).astype(np.float32)
        preds.append(yhat[:, None])
    return np.concatenate(preds, axis=1).astype(np.float32)


# ============================================================
# DRNet / VCNet (Torch SOTA baselines that output curve)
# ============================================================
class DRNet(nn.Module):
    """
    Simple DRNet-style: shared x-representation + multiple heads by dosage bins.
    Use linear interpolation between adjacent bins for smoothness.
    Output: logits if is_binary else regression.
    """
    def __init__(self, x_dim, n_bins=10, hidden=256, depth=3, dropout=0.1, is_binary=False):
        super().__init__()
        self.n_bins = int(n_bins)
        self.is_binary = bool(is_binary)

        layers = []
        d = int(x_dim)
        for _ in range(int(depth)):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.phi = nn.Sequential(*layers)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden + 1, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            ) for _ in range(self.n_bins)
        ])

    def forward(self, x, t):
        # x: (B, x_dim), t: (B,1) in [0,1]
        h = self.phi(x)
        u = torch.clamp(t, 0.0, 1.0) * (self.n_bins - 1)  # (B,1)
        i0 = torch.floor(u).long().squeeze(1)             # (B,)
        i1 = torch.clamp(i0 + 1, max=self.n_bins - 1)     # (B,)
        w1 = (u.squeeze(1) - i0.float()).unsqueeze(1)     # (B,1)
        w0 = 1.0 - w1

        xt = torch.cat([h, t], dim=1)  # (B, hidden+1)

        # gather per-sample head outputs
        y0 = torch.stack([self.heads[i0[j]](xt[j:j+1]) for j in range(len(i0))], dim=0).squeeze(2)  # (B,1)
        y1 = torch.stack([self.heads[i1[j]](xt[j:j+1]) for j in range(len(i1))], dim=0).squeeze(2)  # (B,1)
        y = w0 * y0 + w1 * y1
        return y  # (B,1)


def bspline_basis_1d(t: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """
    Cox篓Cde Boor recursion to compute B-spline basis values.
    Returns: (batch, n_basis) where n_basis = len(knots) - degree - 1
    """
    # t: (B,) or (B,1)
    if t.dim() == 2 and t.size(1) == 1:
        t = t[:, 0]
    t = t.contiguous()

    if not torch.is_tensor(knots):
        knots = torch.tensor(knots, device=t.device, dtype=t.dtype)
    else:
        knots = knots.to(device=t.device, dtype=t.dtype)

    n_knots = int(knots.numel())
    if n_knots < degree + 2:
        raise ValueError(f"Need at least degree+2 knots, got {n_knots} for degree={degree}")

    # 0-degree basis count
    n0 = n_knots - 1  # number of intervals
    B_list = []
    for i in range(n0):
        left = knots[i]
        right = knots[i + 1]
        # include right endpoint only for last interval to cover t==1.0
        if i == n0 - 1:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        B_list.append(mask.to(dtype=t.dtype))
    B = torch.stack(B_list, dim=1)  # (B, n0)

    # recursion: each step reduces basis count by 1
    for k in range(1, int(degree) + 1):
        n_prev = B.size(1)              # = n_knots - k
        n_new = n_prev - 1              # = n_knots - k - 1
        B_new = []
        for i in range(n_new):
            denom1 = (knots[i + k] - knots[i]).item()
            denom2 = (knots[i + k + 1] - knots[i + 1]).item()

            term1 = torch.zeros_like(t)
            term2 = torch.zeros_like(t)

            if denom1 > 1e-12:
                term1 = ((t - knots[i]) / denom1) * B[:, i]
            if denom2 > 1e-12:
                term2 = ((knots[i + k + 1] - t) / denom2) * B[:, i + 1]

            B_new.append(term1 + term2)

        B = torch.stack(B_new, dim=1)   # (B, n_knots-k-1)

    return B  # (B, n_knots-degree-1)


class VCNet(nn.Module):
    """
    VCNet-style: y = sum_k alpha_k(x) * B_k(t)
    alpha(x) produced by an MLP; B(t) are cubic B-spline basis.
    Output: logits if is_binary else regression.
    """
    def __init__(self, x_dim, n_basis=12, hidden=256, depth=3, dropout=0.1, is_binary=False):
        super().__init__()
        self.is_binary = bool(is_binary)
        self.n_basis = int(n_basis)
        self.degree = 3

        layers = []
        d = int(x_dim)
        for _ in range(int(depth)):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(hidden, self.n_basis)]
        self.alpha = nn.Sequential(*layers)

        # open uniform knots in [0,1]
        # n_basis = n_knots - degree - 1 => n_knots = n_basis + degree + 1
        n_knots = self.n_basis + self.degree + 1
        # open uniform: repeat endpoints degree+1 times
        inner = torch.linspace(0, 1, n_knots - 2*(self.degree+1) + 2)
        knots = torch.cat([
            torch.zeros(self.degree+1),
            inner[1:-1],
            torch.ones(self.degree+1)
        ])
        self.register_buffer("knots", knots)

    def forward(self, x, t):
        # x: (B,x_dim), t: (B,1)
        a = self.alpha(x)  # (B, n_basis)
        B = bspline_basis_1d(torch.clamp(t, 0.0, 1.0), self.knots, degree=self.degree)  # (B, n_basis)
        y = torch.sum(a * B, dim=1, keepdim=True)  # (B,1)
        return y


def train_torch_curve_model(model, X, s, y, device, seed=0, is_binary=False,
                            epochs=60, batch=2048, lr=1e-3, wd=1e-5):
    set_all_seeds(int(seed))
    model = model.to(device)

    X = torch.from_numpy(np.asarray(X, np.float32))
    s = torch.from_numpy(np.asarray(s, np.float32)).view(-1, 1)
    y = torch.from_numpy(np.asarray(y, np.float32)).view(-1, 1)

    ds = TensorDataset(X, s, y)
    dl = DataLoader(ds, batch_size=min(int(batch), len(ds)), shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    if is_binary:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        for xb, sb, yb in dl:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb, sb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


@torch.no_grad()
def predict_curve_torch(model, Xte, s_grid, device, is_binary=False):
    Xte_t = torch.from_numpy(np.asarray(Xte, np.float32)).to(device)
    preds = []
    for sv in s_grid:
        st = torch.full((Xte_t.shape[0], 1), float(sv), dtype=torch.float32, device=device)
        out = model(Xte_t, st).squeeze(1)
        if is_binary:
            out = torch.sigmoid(out)
        preds.append(out.detach().cpu().numpy().astype(np.float32)[:, None])
    return np.concatenate(preds, axis=1).astype(np.float32)


# -------------------------
# Synthetic pack per outcome
# -------------------------
def make_synth_pack(df: pd.DataFrame,
                    cov_cols: list,
                    chloride_col: str,
                    y_col: str,
                    s_grid: np.ndarray,
                    rep_id: int,
                    conf_strength: float,
                    seed_base: int = 20260119):
    seed = int(seed_base + 1000 * int(rep_id) + int(conf_strength * 100))
    rng = np.random.default_rng(seed)

    X = df[cov_cols].to_numpy(np.float32)
    chloride_emp = df[chloride_col].to_numpy(np.float64)

    x_scaler = StandardScaler()
    X_std = x_scaler.fit_transform(X).astype(np.float32)

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)

    Xtr_std = X_std[idx_tr]
    Xte_std = X_std[idx_te]
    Xtr = X[idx_tr]
    Xte = X[idx_te]

    kappa = float(10.0 * float(conf_strength))

    dgp = ICU2020_DGP_ExposureOnly(
        X_pool_std=Xtr_std,
        seed_fit=909,
        kappa=kappa,
        epsilon_std=0.15,
        scaling_constant=150.0,
        scaling_offset=50.0,
        treatment_mean=0.60
    )
    dgp.fit()

    s_tr, y_tr_raw, po_tr_raw = dgp.simulate_curves(Xtr_std, s_grid)
    s_te, y_te_raw, po_te_raw = dgp.simulate_curves(Xte_std, s_grid)

    Ttr_chl = quantile_map(s_tr, chloride_emp).astype(np.float32)
    Tte_chl = quantile_map(s_te, chloride_emp).astype(np.float32)
    Tgrid_chl = quantile_map(s_grid, chloride_emp).astype(np.float32)

    y_target = df[y_col].to_numpy(np.float64)
    y_target = y_target[np.isfinite(y_target)]

    is_binary = y_col.endswith("_sign")

    if not is_binary:
        y_tr = monotone_match_distribution(y_tr_raw, y_target)
        y_te = monotone_match_distribution(y_te_raw, y_target)

        base_raw = np.concatenate([y_tr_raw, y_te_raw], axis=0).astype(np.float64)
        qs = np.linspace(0, 1, 2048)
        q_raw = np.quantile(base_raw, qs)
        q_tar = np.quantile(y_target, qs) if y_target.size >= 50 else q_raw

        def f_map(v):
            return np.interp(v, q_raw, q_tar, left=q_tar[0], right=q_tar[-1]).astype(np.float32)

        po_te = np.zeros_like(po_te_raw, dtype=np.float32)
        for j in range(po_te_raw.shape[1]):
            po_te[:, j] = f_map(po_te_raw[:, j])

        return {
            "Xtr": Xtr, "Xte": Xte,
            "s_tr": s_tr.astype(np.float32),
            "s_te": s_te.astype(np.float32),
            "Ttr_chl": Ttr_chl, "Tte_chl": Tte_chl, "Tgrid_chl": Tgrid_chl,
            "y_tr": y_tr.astype(np.float32),
            "y_te": y_te.astype(np.float32),
            "po_te": po_te.astype(np.float32),
            "is_binary": False,
            "kappa": kappa
        }

    else:
        y_real = df[y_col].to_numpy(np.float64)
        y_real = y_real[np.isfinite(y_real)]
        target_prev = float(np.mean(y_real > 0.5)) if y_real.size > 50 else 0.2

        base_raw = np.concatenate([y_tr_raw, y_te_raw], axis=0).astype(np.float64)
        m = float(np.mean(base_raw))
        sdev = float(np.std(base_raw) + 1e-6)

        logits_tr = (y_tr_raw - m) / sdev
        bias = solve_bias_for_prevalence(logits_tr, target_prev=target_prev)

        p_tr = sigmoid(logits_tr + bias).astype(np.float32)
        y_tr = rng.binomial(1, p_tr).astype(np.float32)

        logits_curve = (po_te_raw.astype(np.float64) - m) / sdev
        p_curve = sigmoid(logits_curve + bias).astype(np.float32)

        p_te = sigmoid((y_te_raw - m) / sdev + bias).astype(np.float32)
        y_te = rng.binomial(1, p_te).astype(np.float32)

        return {
            "Xtr": Xtr, "Xte": Xte,
            "s_tr": s_tr.astype(np.float32),
            "s_te": s_te.astype(np.float32),
            "Ttr_chl": Ttr_chl, "Tte_chl": Tte_chl, "Tgrid_chl": Tgrid_chl,
            "y_tr": y_tr.astype(np.float32),
            "y_te": y_te.astype(np.float32),
            "po_te": p_curve.astype(np.float32),
            "is_binary": True,
            "kappa": kappa,
            "target_prev": target_prev
        }


# -------------------------
# Logging (fixed columns!)
# -------------------------
RAW_COLS = [
    "conf_strength", "rep_id", "seed", "y_task", "method",
    "kappa", "mise", "rmse_grid", "auc", "logloss"
]

def append_row_csv(path, row: dict):
    out = {c: row.get(c, np.nan) for c in RAW_COLS}
    df1 = pd.DataFrame([out], columns=RAW_COLS)
    if not os.path.exists(path):
        df1.to_csv(path, index=False)
    else:
        df1.to_csv(path, mode="a", header=False, index=False)

def load_done_keys(raw_csv: str):
    if not os.path.exists(raw_csv):
        return set()
    try:
        df = pd.read_csv(raw_csv)
    except Exception:
        return set()
    need = ["conf_strength", "rep_id", "seed", "y_task", "method"]
    if not all(c in df.columns for c in need):
        return set()
    keys = set()
    for r in df[need].itertuples(index=False):
        keys.add((float(r.conf_strength), int(r.rep_id), int(r.seed), str(r.y_task), str(r.method)))
    return keys


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--t_col", type=str, default="Chloride")
    ap.add_argument("--y_cols", type=str, nargs="+",
                    default=["day28_duration", "day28_sign", "day90_duration", "day90_sign"])

    ap.add_argument("--conf_list", type=float, nargs="+", default=[0.1, 1.0, 5.0])
    ap.add_argument("--rep_ids", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])

    ap.add_argument("--s_grid_size", type=int, default=50)
    ap.add_argument("--n_folds", type=int, default=2)

    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max_n", type=int, default=70000)

    # ORCA config
    ap.add_argument("--orca_nuis_list", type=str, nargs="+", default=["rf", "ridge", "gbdt", "mlp", "xgb"])
    ap.add_argument("--t_modes", type=str, nargs="+", default=["direct", "fourier", "mlp"])
    ap.add_argument("--epochs", type=int, default=60)

    # baselines toggle
    ap.add_argument("--run_bart", action="store_true")

    # improvements for continuous
    ap.add_argument("--orca_stack", action="store_true", help="Enable ORCA-Stack meta learner for *_duration")
    ap.add_argument("--stack_kind", type=str, default="xgb", choices=["xgb", "gbdt", "ridge"])
    ap.add_argument("--y_transform", type=str, default="none", choices=["none", "log1p"],
                    help="Apply to *_duration (y and true curve consistently).")

    # DRNet / VCNet toggles
    ap.add_argument("--run_drnet", action="store_true", help="Enable DRNet baseline")
    ap.add_argument("--run_vcnet", action="store_true", help="Enable VCNet baseline")
    ap.add_argument("--drnet_bins", type=int, default=10)
    ap.add_argument("--vcnet_basis", type=int, default=12)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    out_dir = ensure_dir(args.out_dir)
    cache_dir = ensure_dir(os.path.join(out_dir, "_cache"))
    raw_csv = os.path.join(out_dir, "raw_results.csv")
    sum_csv = os.path.join(out_dir, "summary_results.csv")
    print("OUT:", out_dir)
    print("cache:", cache_dir)
    print("HAS_XGB:", HAS_XGB, "| HAS_BART:", HAS_BART)

    if args.fast:
        args.rep_ids = args.rep_ids[:1]
        args.seeds = args.seeds[:1]
        args.conf_list = args.conf_list[:1]
        args.max_n = min(args.max_n, 25000)
        args.s_grid_size = min(args.s_grid_size, 30)
        args.epochs = min(args.epochs, 30)

    # default: turn on stack in fast runs to help continuous
    if args.fast and not args.orca_stack:
        args.orca_stack = True

    # default: in fast runs, also run drnet/vcnet (they are pretty fast on GPU)
    if args.fast and (not args.run_drnet and not args.run_vcnet):
        args.run_drnet = True
        args.run_vcnet = True

    s_grid = np.linspace(0.0, 1.0, int(args.s_grid_size)).astype(np.float32)

    df, cov_cols = load_and_clean(args.data_path, args.t_col, args.y_cols,
                                 dropna_rows=True, winsor_q=0.01,
                                 max_n=args.max_n, seed=123)
    print(f"Cleaned df: {df.shape}, covariates: {len(cov_cols)}")

    done = load_done_keys(raw_csv)
    print("[RESUME] existing rows:", len(done))

    # plan count
    n_methods_orca = 0
    for nuis in args.orca_nuis_list:
        if nuis.lower() == "xgb" and not HAS_XGB:
            continue
        for _ in args.t_modes:
            n_methods_orca += 1

    base_methods = ["DirectRegression[XGB]", "MLP", "GPS"]
    if args.run_bart and HAS_BART:
        base_methods.append("BART")

    planned = len(args.conf_list) * len(args.rep_ids) * len(args.seeds) * len(args.y_cols) * (n_methods_orca + len(base_methods))
    if args.orca_stack:
        planned += len(args.conf_list) * len(args.rep_ids) * len(args.seeds) * sum([1 for y in args.y_cols if y.endswith("_duration")]) * n_methods_orca
    if args.run_drnet:
        planned += len(args.conf_list) * len(args.rep_ids) * len(args.seeds) * len(args.y_cols)
    if args.run_vcnet:
        planned += len(args.conf_list) * len(args.rep_ids) * len(args.seeds) * len(args.y_cols)
    print("Planned rows:", planned)

    for conf_strength in args.conf_list:
        for rep_id in args.rep_ids:
            for y_task in args.y_cols:
                tag = f"pack_conf{conf_strength}_rep{rep_id}_{y_task}".replace(".", "p")
                npz_path = os.path.join(cache_dir, f"{tag}.npz")

                if os.path.exists(npz_path):
                    z = np.load(npz_path, allow_pickle=True)
                    pack = {k: z[k] for k in z.files}
                    pack["is_binary"] = bool(pack["is_binary"])
                else:
                    pack = make_synth_pack(df, cov_cols, args.t_col, y_task, s_grid,
                                           rep_id=int(rep_id), conf_strength=float(conf_strength))
                    np.savez_compressed(npz_path, **pack)

                Xtr = pack["Xtr"].astype(np.float32)
                Xte = pack["Xte"].astype(np.float32)
                s_tr = pack["s_tr"].astype(np.float32)
                s_te = pack["s_te"].astype(np.float32)
                y_tr = pack["y_tr"].astype(np.float32)
                y_te = pack["y_te"].astype(np.float32)
                po_te = pack["po_te"].astype(np.float32)
                is_binary = bool(pack["is_binary"])
                kappa = float(pack.get("kappa", np.nan))

                # optional transform for duration tasks
                if (not is_binary) and y_task.endswith("_duration"):
                    y_tr = maybe_transform_duration(y_tr, args.y_transform)
                    y_te = maybe_transform_duration(y_te, args.y_transform)
                    po_te = maybe_transform_duration(po_te, args.y_transform)

                print(f"\n[DGP] conf={conf_strength} (kappa={kappa:.2f}) rep={rep_id} y={y_task} binary={is_binary}")
                print("  train n:", len(Xtr), "| test n:", len(Xte))

                for seed in args.seeds:
                    # =====================
                    # Baselines
                    # =====================
                    if HAS_XGB:
                        method = "DirectRegression[XGB]"
                        key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                        if key not in done:
                            pred = baseline_direct_regression(Xtr, s_tr, y_tr, Xte, s_grid,
                                                             model_kind="xgb", is_binary=is_binary, seed=int(seed))
                            row = {
                                "conf_strength": float(conf_strength),
                                "rep_id": int(rep_id),
                                "seed": int(seed),
                                "y_task": y_task,
                                "method": method,
                                "kappa": kappa,
                                "mise": mise_curve(pred, po_te, s_grid),
                                "rmse_grid": rmse_grid(pred, po_te),
                                "auc": np.nan,
                                "logloss": np.nan,
                            }
                            if is_binary:
                                j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                                p_obs = pred[np.arange(len(j)), j]
                                row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                                row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                            append_row_csv(raw_csv, row)
                            done.add(key)
                            print("[OK]", key, "| mise=", row["mise"])

                    method = "MLP"
                    key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                    if key not in done:
                        pred = baseline_mlp_on_xt(Xtr, s_tr, y_tr, Xte, s_grid, is_binary=is_binary, seed=int(seed))
                        row = {
                            "conf_strength": float(conf_strength),
                            "rep_id": int(rep_id),
                            "seed": int(seed),
                            "y_task": y_task,
                            "method": method,
                            "kappa": kappa,
                            "mise": mise_curve(pred, po_te, s_grid),
                            "rmse_grid": rmse_grid(pred, po_te),
                            "auc": np.nan,
                            "logloss": np.nan,
                        }
                        if is_binary:
                            j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                            p_obs = pred[np.arange(len(j)), j]
                            row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                            row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                        append_row_csv(raw_csv, row)
                        done.add(key)
                        print("[OK]", key, "| mise=", row["mise"])

                    method = "GPS"
                    key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                    if key not in done:
                        pred = baseline_gps(Xtr, s_tr, y_tr, Xte, s_grid, is_binary=is_binary, seed=int(seed))
                        row = {
                            "conf_strength": float(conf_strength),
                            "rep_id": int(rep_id),
                            "seed": int(seed),
                            "y_task": y_task,
                            "method": method,
                            "kappa": kappa,
                            "mise": mise_curve(pred, po_te, s_grid),
                            "rmse_grid": rmse_grid(pred, po_te),
                            "auc": np.nan,
                            "logloss": np.nan,
                        }
                        if is_binary:
                            j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                            p_obs = pred[np.arange(len(j)), j]
                            row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                            row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                        append_row_csv(raw_csv, row)
                        done.add(key)
                        print("[OK]", key, "| mise=", row["mise"])

                    if args.run_bart and HAS_BART:
                        method = "BART"
                        key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                        if key not in done:
                            pred = baseline_bart(Xtr, s_tr, y_tr, Xte, s_grid, is_binary=is_binary, seed=int(seed))
                            row = {
                                "conf_strength": float(conf_strength),
                                "rep_id": int(rep_id),
                                "seed": int(seed),
                                "y_task": y_task,
                                "method": method,
                                "kappa": kappa,
                                "mise": mise_curve(pred, po_te, s_grid),
                                "rmse_grid": rmse_grid(pred, po_te),
                                "auc": np.nan,
                                "logloss": np.nan,
                            }
                            if is_binary:
                                j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                                p_obs = pred[np.arange(len(j)), j]
                                row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                                row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                            append_row_csv(raw_csv, row)
                            done.add(key)
                            print("[OK]", key, "| mise=", row["mise"])

                    # =====================
                    # DRNet / VCNet baselines
                    # =====================
                    if args.run_drnet:
                        method = f"DRNet[bins={int(args.drnet_bins)}]"
                        key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                        if key not in done:
                            dr = DRNet(x_dim=Xtr.shape[1], n_bins=int(args.drnet_bins),
                                       hidden=256, depth=3, dropout=0.1, is_binary=is_binary)
                            dr = train_torch_curve_model(
                                dr, Xtr, s_tr, y_tr, device=device, seed=int(seed) + 91001,
                                is_binary=is_binary, epochs=int(args.epochs), batch=4096 if (not args.fast) else 2048,
                                lr=1e-3, wd=1e-5
                            )
                            pred = predict_curve_torch(dr, Xte, s_grid, device=device, is_binary=is_binary)
                            row = {
                                "conf_strength": float(conf_strength),
                                "rep_id": int(rep_id),
                                "seed": int(seed),
                                "y_task": y_task,
                                "method": method,
                                "kappa": kappa,
                                "mise": mise_curve(pred, po_te, s_grid),
                                "rmse_grid": rmse_grid(pred, po_te),
                                "auc": np.nan,
                                "logloss": np.nan,
                            }
                            if is_binary:
                                j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                                p_obs = pred[np.arange(len(j)), j]
                                row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                                row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                            append_row_csv(raw_csv, row)
                            done.add(key)
                            print("[OK]", key, "| mise=", row["mise"])

                    if args.run_vcnet:
                        method = f"VCNet[basis={int(args.vcnet_basis)}]"
                        key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                        if key not in done:
                            vc = VCNet(x_dim=Xtr.shape[1], n_basis=int(args.vcnet_basis),
                                       hidden=256, depth=3, dropout=0.1, is_binary=is_binary)
                            vc = train_torch_curve_model(
                                vc, Xtr, s_tr, y_tr, device=device, seed=int(seed) + 92001,
                                is_binary=is_binary, epochs=int(args.epochs), batch=4096 if (not args.fast) else 2048,
                                lr=1e-3, wd=1e-5
                            )
                            pred = predict_curve_torch(vc, Xte, s_grid, device=device, is_binary=is_binary)
                            row = {
                                "conf_strength": float(conf_strength),
                                "rep_id": int(rep_id),
                                "seed": int(seed),
                                "y_task": y_task,
                                "method": method,
                                "kappa": kappa,
                                "mise": mise_curve(pred, po_te, s_grid),
                                "rmse_grid": rmse_grid(pred, po_te),
                                "auc": np.nan,
                                "logloss": np.nan,
                            }
                            if is_binary:
                                j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                                p_obs = pred[np.arange(len(j)), j]
                                row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                                row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                            append_row_csv(raw_csv, row)
                            done.add(key)
                            print("[OK]", key, "| mise=", row["mise"])

                    # =====================
                    # ORCA grid (+ optional stack for duration)
                    # =====================
                    for nuis in args.orca_nuis_list:
                        if nuis.lower() == "xgb" and not HAS_XGB:
                            continue
                        for tmode in args.t_modes:
                            method = f"ORCA[{nuis.upper()}|t={tmode}]"
                            key = (float(conf_strength), int(rep_id), int(seed), y_task, method)
                            if key not in done:
                                pred = orca_fit_predict_curve(
                                    Xtr, s_tr, y_tr,
                                    Xte, s_grid,
                                    seed=int(seed),
                                    nuis=nuis,
                                    t_mode=tmode,
                                    device=device,
                                    n_folds=int(args.n_folds),
                                    epochs=int(args.epochs),
                                    fast=bool(args.fast),
                                    is_binary=bool(is_binary),
                                    return_oof_factual=False
                                )
                                row = {
                                    "conf_strength": float(conf_strength),
                                    "rep_id": int(rep_id),
                                    "seed": int(seed),
                                    "y_task": y_task,
                                    "method": method,
                                    "kappa": kappa,
                                    "mise": mise_curve(pred, po_te, s_grid),
                                    "rmse_grid": rmse_grid(pred, po_te),
                                    "auc": np.nan,
                                    "logloss": np.nan,
                                }
                                if is_binary:
                                    j = np.argmin(np.abs(s_grid[None, :] - s_te.reshape(-1, 1)), axis=1)
                                    p_obs = pred[np.arange(len(j)), j]
                                    row["auc"] = float(roc_auc_score(y_te, p_obs)) if len(np.unique(y_te)) > 1 else np.nan
                                    row["logloss"] = float(log_loss(y_te, np.clip(p_obs, 1e-6, 1-1e-6)))
                                append_row_csv(raw_csv, row)
                                done.add(key)
                                print("[OK]", key, "| mise=", row["mise"])

                            # ORCA-Stack (continuous only)
                            if (args.orca_stack and (not is_binary) and y_task.endswith("_duration")):
                                stack_method = f"ORCA-Stack[{nuis.upper()}|t={tmode}|meta={args.stack_kind}]"
                                skey = (float(conf_strength), int(rep_id), int(seed), y_task, stack_method)
                                if skey in done:
                                    continue

                                pred_curve, mu_oof = orca_fit_predict_curve(
                                    Xtr, s_tr, y_tr,
                                    Xte, s_grid,
                                    seed=int(seed),
                                    nuis=nuis,
                                    t_mode=tmode,
                                    device=device,
                                    n_folds=int(args.n_folds),
                                    epochs=int(args.epochs),
                                    fast=bool(args.fast),
                                    is_binary=False,
                                    return_oof_factual=True
                                )

                                meta, used_kind = fit_stack_meta(
                                    Xtr, s_tr, mu_oof, y_tr,
                                    seed=int(seed) + 777, kind=args.stack_kind
                                )
                                pred_stack = predict_stack_curve(meta, Xte, s_grid, pred_curve)

                                row = {
                                    "conf_strength": float(conf_strength),
                                    "rep_id": int(rep_id),
                                    "seed": int(seed),
                                    "y_task": y_task,
                                    "method": stack_method,
                                    "kappa": kappa,
                                    "mise": mise_curve(pred_stack, po_te, s_grid),
                                    "rmse_grid": rmse_grid(pred_stack, po_te),
                                    "auc": np.nan,
                                    "logloss": np.nan,
                                }
                                append_row_csv(raw_csv, row)
                                done.add(skey)
                                print("[OK]", skey, "| mise=", row["mise"], "| meta=", used_kind)

    # summary
    dfres = pd.read_csv(raw_csv)
    summ = (dfres.groupby(["conf_strength", "y_task", "method"])
                 .agg(mise_mean=("mise", "mean"),
                      mise_std=("mise", "std"),
                      rmse_mean=("rmse_grid", "mean"),
                      rmse_std=("rmse_grid", "std"),
                      auc_mean=("auc", "mean"),
                      logloss_mean=("logloss", "mean"),
                      count=("mise", "count"))
                 .reset_index())
    summ.to_csv(sum_csv, index=False)

    print("\n========== DONE ==========")
    print("Raw:", raw_csv)
    print("Summary:", sum_csv)
    print("OUT:", out_dir)


if __name__ == "__main__":
    main()
