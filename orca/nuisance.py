import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import set_all_seeds

def _train_val_indices(n: int, seed: int, min_val: int = 16):
    if n < 2:
        raise ValueError("Need at least two samples for nuisance training.")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = min(max(min_val, int(0.2 * n)), n - 1)
    return idx[n_val:], idx[:n_val]


class TorchMLPRegressor:
    def __init__(self, seed: int, hidden=128, lr=2e-3, wd=1e-5, batch=256, epochs=200, patience=20, device="cpu"):
        self.seed = seed
        self.hidden = hidden
        self.lr = lr
        self.wd = wd
        self.batch = batch
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        self.net = None

    def fit(self, X, y):
        set_all_seeds(self.seed)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)

        n = Xs.shape[0]
        tr, va = _train_val_indices(n, self.seed + 77)

        Xtr, ytr = Xs[tr], y[tr]
        Xva, yva = Xs[va], y[va]

        ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        dl = DataLoader(ds, batch_size=min(self.batch, len(ds)), shuffle=True, drop_last=False)

        net = nn.Sequential(
            nn.Linear(Xs.shape[1], self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        ).to(self.device)

        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        mse = nn.MSELoss()

        Xva_t = torch.from_numpy(Xva).to(self.device)
        yva_t = torch.from_numpy(yva).to(self.device)

        best = float("inf"); best_state=None; wait=0
        for _ in range(self.epochs):
            net.train()
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
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
                if wait >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    @torch.no_grad()
    def predict(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        xb = torch.from_numpy(Xs).to(self.device)
        self.net.eval()
        return self.net(xb).squeeze().detach().cpu().numpy().astype(np.float32)


class TorchMLPClassifier:
    def __init__(self, seed: int, hidden=128, lr=2e-3, wd=1e-5, batch=256, epochs=200, patience=20, device="cpu"):
        self.seed = seed
        self.hidden = hidden
        self.lr = lr
        self.wd = wd
        self.batch = batch
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        self.net = None

    def fit(self, X, y01):
        set_all_seeds(self.seed)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        y = y01.astype(np.float32)

        n = Xs.shape[0]
        tr, va = _train_val_indices(n, self.seed + 99)

        Xtr, ytr = Xs[tr], y[tr]
        Xva, yva = Xs[va], y[va]

        ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        dl = DataLoader(ds, batch_size=min(self.batch, len(ds)), shuffle=True, drop_last=False)

        net = nn.Sequential(
            nn.Linear(Xs.shape[1], self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        ).to(self.device)

        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        bce = nn.BCEWithLogitsLoss()

        Xva_t = torch.from_numpy(Xva).to(self.device)
        yva_t = torch.from_numpy(yva).to(self.device)

        best = float("inf"); best_state=None; wait=0
        for _ in range(self.epochs):
            net.train()
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
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
                if wait >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        xb = torch.from_numpy(Xs).to(self.device)
        self.net.eval()
        logit = self.net(xb).squeeze().detach().cpu().numpy().astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-logit))
        p = np.clip(p, 1e-6, 1 - 1e-6).astype(np.float32)
        return np.stack([1 - p, p], axis=1)


def make_nuisance_models(kind: str, seed: int, is_treatment: bool, is_binary: bool, device: str):
    """
    Return a fitted model constructor for:
      e(X): treatment model
      m(X): outcome model

    is_treatment=True means model predicts T (cont or bin).
    is_binary indicates classification model; otherwise regression model.
    """
    k = kind.lower()

    if not is_binary:
        if k == "ridge":
            return Ridge(alpha=1.0)
        if k == "rf":
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=8)
        if k == "gbdt":
            return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
        if k == "mlp":
            return TorchMLPRegressor(seed=seed, device=device)
        raise ValueError(f"Unknown nuisance kind: {kind}")
    else:
        if k == "ridge":
            return LogisticRegression(max_iter=2000)
        if k == "rf":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=8)
        if k == "gbdt":
            return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=seed)
        if k == "mlp":
            return TorchMLPClassifier(seed=seed, device=device)
        raise ValueError(f"Unknown nuisance kind: {kind}")
