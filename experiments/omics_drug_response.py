"""
Pharmacogenomic drug-response benchmark for ORCA.

This script is intended for the Bioinformatics-targeted manuscript revision.
It evaluates factual drug-response prediction on held-out cell lines using
omics features and drug descriptors. ORCA is implemented here as a vector
treatment residualization model:

    cell omics X, drug descriptor T, response Y
    m(X) predicts baseline response
    e(X) predicts expected drug descriptor
    residual drug descriptor = T - e(X)
    residual network predicts Y - m(X) from [X, T, T - e(X)]

Generic CSV requirements:
  --omics_csv: one row per cell line, with a cell-line id column and numeric omics columns
  --response_csv: one row per cell line-drug pair, with cell-line id, drug id/SMILES, and response

Example:
  python experiments/omics_drug_response.py \
    --omics_csv data/ccle_expression.csv \
    --response_csv data/gdsc_response.csv \
    --cell_col DepMap_ID \
    --response_cell_col DepMap_ID \
    --drug_col smiles \
    --y_col LN_IC50 \
    --out_dir runs/gdsc_orca
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    if np.std(y_true[mask]) < 1e-12 or np.std(y_pred[mask]) < 1e-12:
        return float("nan")
    return float(fn(y_true[mask], y_pred[mask])[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": safe_corr(pearsonr, y_true, y_pred),
        "spearman": safe_corr(spearmanr, y_true, y_pred),
    }


@dataclass
class Config:
    omics_csv: Optional[str]
    response_csv: Optional[str]
    cell_col: str
    response_cell_col: Optional[str]
    drug_col: str
    y_col: str
    drug_descriptor_csv: Optional[str]
    drug_descriptor_col: Optional[str]
    tdc_dataset: Optional[str]
    tdc_path: Optional[str]
    out_dir: str
    seed: int
    test_size: float
    val_size: float
    drug_hash_dim: int
    max_omics_features: int
    variance_threshold: float
    hidden_dim: int
    depth: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    device: str
    run_sklearn_baselines: bool
    max_rows: Optional[int]
    disable_residual_t: bool
    split_seed: Optional[int]
    e_alpha: float
    residual_t_scale: float
    orca_arch: str


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    cell_ids: np.ndarray
    drug_ids: np.ndarray
    x_cols: List[str]
    t_cols: List[str]


def numeric_columns(df: pd.DataFrame, exclude: set[str]) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def load_generic_csv(args: Config) -> DatasetBundle:
    if not args.omics_csv or not args.response_csv:
        raise ValueError("--omics_csv and --response_csv are required unless --tdc_dataset is used.")

    omics = pd.read_csv(args.omics_csv)
    response = pd.read_csv(args.response_csv)

    response_cell_col = args.response_cell_col or args.cell_col
    required_omics = {args.cell_col}
    required_response = {response_cell_col, args.drug_col, args.y_col}
    missing_omics = required_omics - set(omics.columns)
    missing_response = required_response - set(response.columns)
    if missing_omics:
        raise ValueError(f"Missing omics columns: {sorted(missing_omics)}")
    if missing_response:
        raise ValueError(f"Missing response columns: {sorted(missing_response)}")

    x_cols = numeric_columns(omics, exclude={args.cell_col})
    if not x_cols:
        raise ValueError("No numeric omics columns found.")

    if args.max_omics_features and len(x_cols) > args.max_omics_features:
        variances = omics[x_cols].var(axis=0, numeric_only=True).sort_values(ascending=False)
        x_cols = variances.index[: args.max_omics_features].tolist()

    omics = omics[[args.cell_col] + x_cols].drop_duplicates(args.cell_col)
    response = response[[response_cell_col, args.drug_col, args.y_col]].dropna()
    response = response.rename(columns={response_cell_col: args.cell_col})

    frame = response.merge(omics, on=args.cell_col, how="inner")
    if args.max_rows:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=args.seed)
    if frame.empty:
        raise ValueError("No rows remain after merging omics and response data.")

    if args.drug_descriptor_csv:
        drug_key = args.drug_descriptor_col or args.drug_col
        descriptors = pd.read_csv(args.drug_descriptor_csv)
        if drug_key not in descriptors.columns:
            raise ValueError(f"Missing drug descriptor key column: {drug_key}")
        desc_cols = numeric_columns(descriptors, exclude={drug_key})
        if not desc_cols:
            raise ValueError("No numeric drug descriptor columns found.")
        descriptors = descriptors[[drug_key] + desc_cols].drop_duplicates(drug_key)
        descriptors = descriptors.rename(columns={drug_key: args.drug_col})
        frame = frame.merge(descriptors, on=args.drug_col, how="inner")
        t = frame[desc_cols].astype(np.float32).to_numpy()
        t_cols = desc_cols
    else:
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=args.drug_hash_dim,
            alternate_sign=False,
            norm=None,
            lowercase=False,
        )
        t = vectorizer.transform(frame[args.drug_col].astype(str)).astype(np.float32).toarray()
        t_cols = [f"drug_hash_{i}" for i in range(args.drug_hash_dim)]

    x = frame[x_cols].astype(np.float32).to_numpy()
    y = frame[args.y_col].astype(np.float32).to_numpy()
    cell_ids = frame[args.cell_col].astype(str).to_numpy()
    drug_ids = frame[args.drug_col].astype(str).to_numpy()

    keep = np.isfinite(x).all(axis=1) & np.isfinite(t).all(axis=1) & np.isfinite(y)
    frame = frame.loc[keep].reset_index(drop=True)
    return DatasetBundle(
        frame=frame,
        x=x[keep],
        t=t[keep],
        y=y[keep],
        cell_ids=cell_ids[keep],
        drug_ids=drug_ids[keep],
        x_cols=x_cols,
        t_cols=t_cols,
    )


def load_tdc_dataset(args: Config) -> DatasetBundle:
    """Best-effort PyTDC loader.

    PyTDC drug-response datasets differ in column naming and usually do not
    include full omics matrices in the same table. For manuscript-grade results,
    the generic CSV route is recommended after exporting a clean joined table.
    """
    try:
        from tdc.multi_pred import DrugRes  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTDC is not installed. Install it or use the generic CSV workflow.") from exc

    if not args.tdc_dataset:
        raise ValueError("--tdc_dataset is required for the PyTDC loader.")

    data = DrugRes(name=args.tdc_dataset, path=args.tdc_path)
    df = data.get_data()
    out = ensure_dir(args.out_dir)
    raw_path = out / f"tdc_{args.tdc_dataset}_raw.csv"
    df.to_csv(raw_path, index=False)
    raise RuntimeError(
        "PyTDC data were downloaded, but this script needs an explicit omics matrix "
        "for held-out cell-line evaluation. Exported the raw PyTDC table to "
        f"{raw_path}. Join it with omics features and rerun with --omics_csv/--response_csv."
    )


def split_by_cell(
    cell_ids: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.unique(cell_ids)
    if len(cells) < 5:
        raise ValueError("Need at least five unique cell lines for a held-out split.")

    trainval_cells, test_cells = train_test_split(
        cells,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    adjusted_val = val_size / max(1e-8, 1.0 - test_size)
    adjusted_val = min(max(adjusted_val, 0.05), 0.5)
    train_cells, val_cells = train_test_split(
        trainval_cells,
        test_size=adjusted_val,
        random_state=seed + 1,
        shuffle=True,
    )

    train_idx = np.flatnonzero(np.isin(cell_ids, train_cells))
    val_idx = np.flatnonzero(np.isin(cell_ids, val_cells))
    test_idx = np.flatnonzero(np.isin(cell_ids, test_cells))
    return train_idx, val_idx, test_idx


class Preprocessor:
    def __init__(self, max_omics_features: int, variance_threshold: float):
        self.max_omics_features = max_omics_features
        self.variance_threshold = variance_threshold
        self.selector: Optional[VarianceThreshold] = None
        self.selected_idx: Optional[np.ndarray] = None
        self.x_scaler = StandardScaler()
        self.t_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> "Preprocessor":
        selector = VarianceThreshold(threshold=self.variance_threshold)
        x_v = selector.fit_transform(x)
        support = np.flatnonzero(selector.get_support())
        if self.max_omics_features and x_v.shape[1] > self.max_omics_features:
            variances = np.var(x_v, axis=0)
            top_local = np.argsort(variances)[-self.max_omics_features :]
            top_local = np.sort(top_local)
            self.selected_idx = support[top_local]
        else:
            self.selected_idx = support
        self.selector = selector
        self.x_scaler.fit(x[:, self.selected_idx])
        self.t_scaler.fit(t)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.selected_idx is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        return self.x_scaler.transform(x[:, self.selected_idx]).astype(np.float32)

    def transform_t(self, t: np.ndarray) -> np.ndarray:
        return self.t_scaler.transform(t).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


class ResidualNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class TwoTowerResidualNet(nn.Module):
    """Two-tower residual network for expression and drug-descriptor inputs.

    The network first encodes omics covariates and drug descriptors through
    separate towers, then fuses the learned representations with the residualized
    drug descriptor. This matches the pharmacogenomic ORCA variant reported in
    the Bioinformatics submission.
    """

    def __init__(
        self,
        x_dim: int,
        t_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        use_residual_t: bool,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.t_dim = t_dim
        self.use_residual_t = use_residual_t
        self.x_tower = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.t_tower = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        fusion_dim = 2 * hidden_dim + (t_dim if use_residual_t else 0)
        layers: List[nn.Module] = []
        dim = fusion_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, : self.x_dim]
        t = z[:, self.x_dim : self.x_dim + self.t_dim]
        hx = self.x_tower(x)
        ht = self.t_tower(t)
        if self.use_residual_t:
            rt = z[:, self.x_dim + self.t_dim :]
            fused = torch.cat([hx, ht, rt], dim=1)
        else:
            fused = torch.cat([hx, ht], dim=1)
        return self.head(fused).squeeze(-1)


def batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def fit_orca(
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    t_test: np.ndarray,
    args: Config,
) -> Tuple[np.ndarray, Dict[str, float]]:
    m_model = MLPRegressor(
        hidden_layer_sizes=(args.hidden_dim, args.hidden_dim),
        activation="relu",
        alpha=args.weight_decay,
        learning_rate_init=args.lr,
        max_iter=500,
        early_stopping=True,
        random_state=args.seed,
    )
    e_model = Ridge(alpha=args.e_alpha, random_state=args.seed)

    m_model.fit(x_train, y_train)
    e_model.fit(x_train, t_train)

    m_train = m_model.predict(x_train).astype(np.float32)
    m_val = m_model.predict(x_val).astype(np.float32)
    m_test = m_model.predict(x_test).astype(np.float32)
    e_train = e_model.predict(x_train).astype(np.float32)
    e_val = e_model.predict(x_val).astype(np.float32)
    e_test = e_model.predict(x_test).astype(np.float32)

    rt_train = (t_train - e_train) * args.residual_t_scale
    rt_val = (t_val - e_val) * args.residual_t_scale
    rt_test = (t_test - e_test) * args.residual_t_scale
    ry_train = y_train - m_train
    ry_val = y_val - m_val

    if args.disable_residual_t:
        z_train = np.concatenate([x_train, t_train], axis=1).astype(np.float32)
        z_val = np.concatenate([x_val, t_val], axis=1).astype(np.float32)
        z_test = np.concatenate([x_test, t_test], axis=1).astype(np.float32)
    else:
        z_train = np.concatenate([x_train, t_train, rt_train], axis=1).astype(np.float32)
        z_val = np.concatenate([x_val, t_val, rt_val], axis=1).astype(np.float32)
        z_test = np.concatenate([x_test, t_test, rt_test], axis=1).astype(np.float32)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.orca_arch == "plain":
        model = ResidualNet(z_train.shape[1], args.hidden_dim, args.depth, args.dropout).to(device)
    elif args.orca_arch == "twotower":
        model = TwoTowerResidualNet(
            x_dim=x_train.shape[1],
            t_dim=t_train.shape[1],
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            use_residual_t=not args.disable_residual_t,
        ).to(device)
    else:
        raise ValueError(f"Unknown ORCA architecture: {args.orca_arch}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(args.seed)

    z_train_t = torch.from_numpy(z_train).to(device)
    ry_train_t = torch.from_numpy(ry_train.astype(np.float32)).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)
    ry_val_t = torch.from_numpy(ry_val.astype(np.float32)).to(device)

    best_state = None
    best_val = float("inf")
    stale = 0
    history: Dict[str, float] = {}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for idx in batches(len(z_train), args.batch_size, shuffle=True, rng=rng):
            opt.zero_grad(set_to_none=True)
            pred = model(z_train_t[idx])
            loss = loss_fn(pred, ry_train_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(z_val_t), ry_val_t).detach().cpu())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        history = {
            "orca_best_val_residual_mse": best_val,
            "orca_last_train_residual_mse": float(np.mean(train_losses)),
            "orca_epochs_ran": float(epoch),
        }
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        residual_pred = model(torch.from_numpy(z_test).to(device)).detach().cpu().numpy()
    y_pred = m_test + residual_pred
    history["orca_training_seconds"] = time.time() - start_time
    history["orca_device"] = 1.0 if device.type == "cuda" else 0.0
    history["orca_arch_twotower"] = 1.0 if args.orca_arch == "twotower" else 0.0
    return y_pred.astype(np.float32), history


def fit_sklearn_baselines(
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    t_test: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    z_train = np.concatenate([x_train, t_train], axis=1)
    z_test = np.concatenate([x_test, t_test], axis=1)
    models = {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "elastic_net": ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=10000, random_state=seed),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=512,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=0.01,
            random_state=seed,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=8,
            verbose=-1,
        ),
    }
    preds: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(z_train, y_train)
        preds[name] = model.predict(z_test).astype(np.float32)
    return preds


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run(args: Config) -> None:
    set_all_seeds(args.seed)
    out = ensure_dir(args.out_dir)

    if args.tdc_dataset:
        bundle = load_tdc_dataset(args)
    else:
        bundle = load_generic_csv(args)

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    train_idx, val_idx, test_idx = split_by_cell(bundle.cell_ids, split_seed, args.test_size, args.val_size)
    prep = Preprocessor(args.max_omics_features, args.variance_threshold).fit(
        bundle.x[train_idx],
        bundle.t[train_idx],
        bundle.y[train_idx],
    )

    x_train = prep.transform_x(bundle.x[train_idx])
    x_val = prep.transform_x(bundle.x[val_idx])
    x_test = prep.transform_x(bundle.x[test_idx])
    t_train = prep.transform_t(bundle.t[train_idx])
    t_val = prep.transform_t(bundle.t[val_idx])
    t_test = prep.transform_t(bundle.t[test_idx])
    y_train = prep.transform_y(bundle.y[train_idx])
    y_val = prep.transform_y(bundle.y[val_idx])
    y_test = prep.transform_y(bundle.y[test_idx])

    y_true = bundle.y[test_idx]
    predictions: Dict[str, np.ndarray] = {}
    fit_info: Dict[str, float] = {}

    y_orca_scaled, orca_info = fit_orca(
        x_train,
        t_train,
        y_train,
        x_val,
        t_val,
        y_val,
        x_test,
        t_test,
        args,
    )
    orca_name = "orca_no_residual_t" if args.disable_residual_t else "orca"
    predictions[orca_name] = prep.inverse_y(y_orca_scaled)
    fit_info.update(orca_info)

    if args.run_sklearn_baselines:
        baseline_scaled = fit_sklearn_baselines(x_train, t_train, y_train, x_test, t_test, args.seed)
        for name, pred_scaled in baseline_scaled.items():
            predictions[name] = prep.inverse_y(pred_scaled)

    metrics = {name: regression_metrics(y_true, pred) for name, pred in predictions.items()}

    pred_frame = pd.DataFrame(
        {
            "cell_id": bundle.cell_ids[test_idx],
            "drug_id": bundle.drug_ids[test_idx],
            "y_true": y_true,
        }
    )
    for name, pred in predictions.items():
        pred_frame[f"y_pred_{name}"] = pred
    pred_frame.to_csv(out / "predictions_test.csv", index=False)

    split_info = {
        "n_rows": int(len(bundle.y)),
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_cells": int(len(np.unique(bundle.cell_ids))),
        "n_train_cells": int(len(np.unique(bundle.cell_ids[train_idx]))),
        "n_val_cells": int(len(np.unique(bundle.cell_ids[val_idx]))),
        "n_test_cells": int(len(np.unique(bundle.cell_ids[test_idx]))),
        "n_drugs_total": int(len(np.unique(bundle.drug_ids))),
        "n_omics_features_raw": int(bundle.x.shape[1]),
        "n_omics_features_used": int(x_train.shape[1]),
        "n_drug_features": int(t_train.shape[1]),
        "test_cells": sorted(np.unique(bundle.cell_ids[test_idx]).tolist()),
        "split_seed": int(split_seed),
    }
    write_json(out / "metrics.json", metrics)
    write_json(out / "fit_info.json", fit_info)
    write_json(out / "split_info.json", split_info)
    write_json(out / "config.json", asdict(args))

    leaderboard = pd.DataFrame(metrics).T.sort_values("rmse")
    leaderboard.to_csv(out / "leaderboard.csv")
    print("\nHeld-out cell-line results")
    print(leaderboard.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to: {out}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omics_csv", default=None)
    parser.add_argument("--response_csv", default=None)
    parser.add_argument("--cell_col", default="cell_id")
    parser.add_argument("--response_cell_col", default=None)
    parser.add_argument("--drug_col", default="drug_id")
    parser.add_argument("--y_col", default="response")
    parser.add_argument("--drug_descriptor_csv", default=None)
    parser.add_argument("--drug_descriptor_col", default=None)
    parser.add_argument("--tdc_dataset", default=None)
    parser.add_argument("--tdc_path", default=None)
    parser.add_argument("--out_dir", default="runs/omics_drug_response")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--drug_hash_dim", type=int, default=256)
    parser.add_argument("--max_omics_features", type=int, default=5000)
    parser.add_argument("--variance_threshold", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_sklearn_baselines", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--disable_residual_t", action="store_true")
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--e_alpha", type=float, default=1.0)
    parser.add_argument("--residual_t_scale", type=float, default=1.0)
    parser.add_argument("--orca_arch", choices=["plain", "twotower"], default="plain")
    ns = parser.parse_args()
    return Config(
        omics_csv=ns.omics_csv,
        response_csv=ns.response_csv,
        cell_col=ns.cell_col,
        response_cell_col=ns.response_cell_col,
        drug_col=ns.drug_col,
        y_col=ns.y_col,
        drug_descriptor_csv=ns.drug_descriptor_csv,
        drug_descriptor_col=ns.drug_descriptor_col,
        tdc_dataset=ns.tdc_dataset,
        tdc_path=ns.tdc_path,
        out_dir=ns.out_dir,
        seed=ns.seed,
        test_size=ns.test_size,
        val_size=ns.val_size,
        drug_hash_dim=ns.drug_hash_dim,
        max_omics_features=ns.max_omics_features,
        variance_threshold=ns.variance_threshold,
        hidden_dim=ns.hidden_dim,
        depth=ns.depth,
        dropout=ns.dropout,
        lr=ns.lr,
        weight_decay=ns.weight_decay,
        batch_size=ns.batch_size,
        epochs=ns.epochs,
        patience=ns.patience,
        device=ns.device,
        run_sklearn_baselines=not ns.no_sklearn_baselines,
        max_rows=ns.max_rows,
        disable_residual_t=ns.disable_residual_t,
        split_seed=ns.split_seed,
        e_alpha=ns.e_alpha,
        residual_t_scale=ns.residual_t_scale,
        orca_arch=ns.orca_arch,
    )


if __name__ == "__main__":
    run(parse_args())
