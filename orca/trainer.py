import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

from .nuisance import make_nuisance_models
from .utils import set_all_seeds
from .metrics import mise_grid, rmse, sqrt_pehe

def fit_orca(dataset, cfg, seed: int = 0, device: str = "cuda"):
    """
    Returns fitted ORCA instance and a metrics dict (if truth available).
    """
    from .model import ORCA

    set_all_seeds(seed)

    Xtr = dataset.X_train
    Ttr = dataset.T_train
    Ytr = dataset.Y_train
    Xte = dataset.X_test

    # build t_grid for contT
    t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.t_grid_size).astype(np.float32)

    model = ORCA(cfg, device=device)

    is_contT = cfg.task.startswith("contT_")
    is_binT = cfg.task.startswith("binT_")
    is_binY = cfg.task.endswith("_binY")

    # cross-fitting wrapper
    K = int(cfg.crossfit_folds)
    if K <= 1:
        m_model = make_nuisance_models(cfg.nuisance, seed + 10, is_treatment=False, is_binary=is_binY, device=device)
        e_model = make_nuisance_models(cfg.nuisance, seed + 20, is_treatment=True,  is_binary=is_binT, device=device)
        model.fit_onefold(Xtr, Ttr, Ytr, m_model=m_model, e_model=e_model, seed=seed + 999)
        pred = _predict(model, Xte, cfg, t_grid)
    else:
        # average over folds
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        preds = []
        for fold_id, (idx_fit, idx_res) in enumerate(kf.split(Xtr)):
            X_fit, T_fit, Y_fit = Xtr[idx_fit], Ttr[idx_fit], Ytr[idx_fit]
            # fit nuisances on fit, train residual net on res (simple crossfit style)
            m_model = make_nuisance_models(cfg.nuisance, seed + 10 + fold_id, is_treatment=False, is_binary=is_binY, device=device)
            e_model = make_nuisance_models(cfg.nuisance, seed + 20 + fold_id, is_treatment=True,  is_binary=is_binT, device=device)

            # NOTE: skeleton uses res subset to train net; for your final open-source, you can refine.
            m_model.fit(X_fit, Y_fit if not is_binY else Y_fit.astype(int))
            if is_contT:
                e_model.fit(X_fit, T_fit)
            else:
                e_model.fit(X_fit, T_fit.astype(int))

            # train net on res subset but keep nuisance from fit
            model_fold = ORCA(cfg, device=device)
            model_fold._build_net(Xtr.shape[1])
            model_fold.m_model = m_model
            model_fold.e_model = e_model
            model_fold.fit_onefold(Xtr[idx_res], Ttr[idx_res], Ytr[idx_res],
                                   m_model=m_model, e_model=e_model, seed=seed + 999 + fold_id)

            preds.append(_predict(model_fold, Xte, cfg, t_grid))

        pred = np.mean(np.stack(preds, axis=0), axis=0)
        model = model_fold  # return last fold model (for simplicity)

    metrics = _evaluate(dataset, cfg, pred, t_grid)
    return model, metrics


def _predict(model, Xte, cfg, t_grid):
    if cfg.task.startswith("contT_"):
        return model.predict_contT_grid(Xte, t_grid)
    else:
        mu0, mu1 = model.predict_binT(Xte)
        return np.stack([mu0, mu1], axis=1)


def _evaluate(dataset, cfg, pred, t_grid):
    """
    Compute metrics if truth exists in dataset.
    """
    out = {}

    if cfg.task.startswith("contT_"):
        if dataset.mu_grid_test is None:
            return out
        out["mise"] = mise_grid(pred, dataset.mu_grid_test, t_grid)
        out["rmse_grid"] = rmse(pred, dataset.mu_grid_test)
        return out

    # binT tasks
    if dataset.mu0_test is None or dataset.mu1_test is None:
        return out

    mu0_hat = pred[:, 0].astype(np.float32)
    mu1_hat = pred[:, 1].astype(np.float32)
    ite_hat = (mu1_hat - mu0_hat).astype(np.float32)

    ite_true = (dataset.mu1_test - dataset.mu0_test).astype(np.float32)
    out["sqrt_pehe"] = sqrt_pehe(ite_hat, ite_true)
    out["ate_err"] = abs(float(np.mean(ite_hat)) - float(np.mean(ite_true)))
    return out
