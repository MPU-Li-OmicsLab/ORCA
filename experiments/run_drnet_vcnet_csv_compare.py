# -*- coding: utf-8 -*-
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# utils
# -------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def rmse(yhat, y):
    yhat=np.asarray(yhat).reshape(-1)
    y=np.asarray(y).reshape(-1)
    return float(np.sqrt(np.mean((yhat-y)**2)))

def mae(yhat, y):
    yhat=np.asarray(yhat).reshape(-1)
    y=np.asarray(y).reshape(-1)
    return float(np.mean(np.abs(yhat-y)))

def make_t_grid(t, qmin, qmax, G):
    lo=float(np.quantile(t, qmin))
    hi=float(np.quantile(t, qmax))
    grid=np.linspace(lo, hi, G).astype("float32")
    return grid

def interp_grid_to_tobs(mu_grid, t_grid, t_obs):
    """
    mu_grid: (n,G) predicted at t_grid
    return: (n,) predicted at each row's t_obs using linear interpolation
    """
    t_grid=np.asarray(t_grid, dtype=float)
    t_obs=np.asarray(t_obs, dtype=float)
    mu_grid=np.asarray(mu_grid, dtype=float)
    n,G=mu_grid.shape
    # clip to grid range
    to=np.clip(t_obs, t_grid[0], t_grid[-1])
    # find right index
    j=np.searchsorted(t_grid, to, side="right")
    j=np.clip(j, 1, G-1)
    j0=j-1; j1=j
    t0=t_grid[j0]; t1=t_grid[j1]
    w=(to-t0)/(t1-t0+1e-12)
    y0=mu_grid[np.arange(n), j0]
    y1=mu_grid[np.arange(n), j1]
    return (1-w)*y0 + w*y1

class StandardScaler:
    def __init__(self, eps=1e-8):
        self.mean=None; self.std=None; self.eps=eps
    def fit(self, X):
        self.mean=X.mean(0, keepdims=True)
        self.std=np.maximum(X.std(0, keepdims=True), self.eps)
    def transform(self, X):
        return (X-self.mean)/self.std
    def fit_transform(self, X):
        self.fit(X); return self.transform(X)

# -------------------------
# torch models
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=3, out_dim=128, dropout=0.1):
        super().__init__()
        layers=[]
        d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d=hidden
        layers += [nn.Linear(d, out_dim)]
        self.net=nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def fourier_feats(t, n_freq=10):
    # t: (n,1)
    x=t * np.pi
    feats=[torch.ones_like(x)]
    for k in range(1, n_freq+1):
        feats.append(torch.sin(k*x))
        feats.append(torch.cos(k*x))
    return torch.cat(feats, dim=1)  # (n, 1+2*n_freq)

class SLearner(nn.Module):
    def __init__(self, x_dim, hidden=256, depth=3, dropout=0.1, n_freq=10):
        super().__init__()
        self.n_freq=n_freq
        self.rep=MLP(x_dim + (1+2*n_freq), hidden=hidden, depth=depth, out_dim=hidden, dropout=dropout)
        self.out=nn.Linear(hidden, 1)
    def forward(self, x, t):
        tf=fourier_feats(t, self.n_freq)
        h=self.rep(torch.cat([x, tf], dim=1))
        return self.out(h)

class VCNetLike(nn.Module):
    """
    y = m(x) + w(x)^T b(t)
    b(t): Fourier basis
    """
    def __init__(self, x_dim, hidden=256, depth=3, dropout=0.1, n_freq=10):
        super().__init__()
        self.n_freq=n_freq
        self.rep=MLP(x_dim, hidden=hidden, depth=depth, out_dim=hidden, dropout=dropout)
        self.m=nn.Linear(hidden, 1)
        self.w=nn.Linear(hidden, 1+2*n_freq)  # coefficients for basis
    def forward(self, x, t):
        h=self.rep(x)
        m=self.m(h)                          # (n,1)
        w=self.w(h)                          # (n,B)
        b=fourier_feats(t, self.n_freq)      # (n,B)
        y=m + (w*b).sum(dim=1, keepdim=True)
        return y

class DRNetLike(nn.Module):
    """
    Discretize t into K bins; shared rep(x); head_k takes [rep, b(t)].
    """
    def __init__(self, x_dim, K=10, hidden=256, depth=3, dropout=0.1, n_freq=10):
        super().__init__()
        self.K=K; self.n_freq=n_freq
        self.rep=MLP(x_dim, hidden=hidden, depth=depth, out_dim=hidden, dropout=dropout)
        B=1+2*n_freq
        self.heads=nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden+B, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1)
            ) for _ in range(K)
        ])
    def forward(self, x, t, k_idx):
        h=self.rep(x)
        b=fourier_feats(t, self.n_freq)
        inp=torch.cat([h,b], dim=1)
        yhat=torch.zeros((x.shape[0],1), device=x.device)
        for k in range(self.K):
            mask=(k_idx==k)
            if mask.any():
                yhat[mask]=self.heads[k](inp[mask])
        return yhat

# -------------------------
# training
# -------------------------
def train_eval(model_name, model, Xtr, Ttr, Ytr, Xte, Tte, Yte, *,
               t_grid, device, lr=1e-3, wd=1e-5, batch=1024, epochs=200, patience=15, K=10, t_bins=None):
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n=len(Xtr)
    idx=np.arange(n)
    np.random.shuffle(idx)
    n_val=max(128, int(0.1*n))
    va=idx[:n_val]; tr=idx[n_val:]

    Xtr_t=torch.from_numpy(Xtr[tr]).float().to(device)
    Ttr_t=torch.from_numpy(Ttr[tr]).float().reshape(-1,1).to(device)
    Ytr_t=torch.from_numpy(Ytr[tr]).float().reshape(-1,1).to(device)

    Xva_t=torch.from_numpy(Xtr[va]).float().to(device)
    Tva_t=torch.from_numpy(Ttr[va]).float().reshape(-1,1).to(device)
    Yva_t=torch.from_numpy(Ytr[va]).float().reshape(-1,1).to(device)

    best=float("inf"); best_state=None; bad=0

    def drnet_bin(t):
        # t_bins: array of edges length K+1
        j=np.searchsorted(t_bins, t, side="right")-1
        return np.clip(j, 0, K-1).astype(np.int64)

    for ep in range(epochs):
        model.train()
        # minibatches
        perm=np.random.permutation(len(tr))
        for s in range(0, len(tr), batch):
            ii=perm[s:s+batch]
            xb=Xtr_t[ii]; tb=Ttr_t[ii]; yb=Ytr_t[ii]
            opt.zero_grad(set_to_none=True)
            if model_name=="drnet":
                k_idx=torch.from_numpy(drnet_bin(Ttr[tr][ii])).to(device)
                pred=model(xb, tb, k_idx)
            else:
                pred=model(xb, tb)
            loss=F.mse_loss(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if model_name=="drnet":
                k_idx=torch.from_numpy(drnet_bin(Ttr[va])).to(device)
                v=F.mse_loss(model(Xva_t, Tva_t, k_idx), Yva_t).item()
            else:
                v=F.mse_loss(model(Xva_t, Tva_t), Yva_t).item()

        if v < best - 1e-5:
            best=v
            best_state={k:vv.detach().cpu().clone() for k,vv in model.state_dict().items()}
            bad=0
        else:
            bad += 1
            if bad>=patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict on grid (for visualization/ADRF)
    model.eval()
    Xte_t=torch.from_numpy(Xte).float().to(device)
    n_te=len(Xte)
    G=len(t_grid)
    mu_hat=np.zeros((n_te, G), dtype=np.float32)

    def drnet_bin_np(tvals):
        j=np.searchsorted(t_bins, tvals, side="right")-1
        return np.clip(j, 0, K-1).astype(np.int64)

    with torch.no_grad():
        for j, tv in enumerate(t_grid):
            tb=torch.full((n_te,1), float(tv), device=device)
            if model_name=="drnet":
                k_idx=torch.from_numpy(drnet_bin_np(np.full(n_te, float(tv)))).to(device)
                yhat=model(Xte_t, tb, k_idx).squeeze().cpu().numpy().astype(np.float32)
            else:
                yhat=model(Xte_t, tb).squeeze().cpu().numpy().astype(np.float32)
            mu_hat[:,j]=yhat

    # observed-point prediction by interpolation
    yhat_obs=interp_grid_to_tobs(mu_hat, t_grid, Tte)
    return {
        "rmse": rmse(yhat_obs, Yte),
        "mae": mae(yhat_obs, Yte),
        "mu_hat_grid": mu_hat,
    }

def load_csv_numeric(data_path, t_col, y_col, drop_other_y=False, impute=True):
    df=pd.read_csv(data_path)

    # robust numeric conversion: blanks->NaN, strings->NaN
    def to_num(s):
        return pd.to_numeric(s.replace(r'^\s*$', np.nan, regex=True), errors="coerce")

    y=to_num(df[y_col])
    t=to_num(df[t_col])

    # feature cols
    feat_cols=[c for c in df.columns if c not in [t_col, y_col]]
    if drop_other_y:
        import re as _re
        feat_cols=[c for c in feat_cols if not _re.search(r"day\d+_duration", c)]
    X=df[feat_cols].copy()
    for c in X.columns:
        X[c]=to_num(X[c])

    # drop rows missing y or t
    m=~(y.isna()|t.isna())
    X=X[m].values.astype(np.float32)
    t=t[m].values.astype(np.float32)
    y=y[m].values.astype(np.float32)

    # impute features
    if impute:
        col_med=np.nanmedian(X, axis=0)
        inds=np.where(np.isnan(X))
        X[inds]=col_med[inds[1]]

    return X, t, y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--t_col", type=str, required=True)
    ap.add_argument("--y_col", type=str, required=True)
    ap.add_argument("--drop_other_y", action="store_true")
    ap.add_argument("--out_dir", type=str, default="/nas/zrp_data/cont_runs")
    ap.add_argument("--tag", type=str, default="csv_drnet_vcnet_compare")

    ap.add_argument("--seed_list", type=str, default="100,101,102")
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--grid_size", type=int, default=64)
    ap.add_argument("--t_qmin", type=float, default=0.01)
    ap.add_argument("--t_qmax", type=float, default=0.99)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n_freq", type=int, default=10)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)

    ap.add_argument("--drnet_bins", type=int, default=10)

    ap.add_argument("--orca_pred", type=str, default="")  # optional: path to ORCA pred.npz
    args=ap.parse_args()

    seeds=[int(x) for x in args.seed_list.split(",") if x.strip()]
    os.makedirs(os.path.join(args.out_dir, args.tag), exist_ok=True)

    device="cuda" if torch.cuda.is_available() else "cpu"

    X, t, y = load_csv_numeric(args.data_path, args.t_col, args.y_col, impute=True, drop_other_y=args.drop_other_y)

    # global t_grid (same for all seeds)
    t_grid=make_t_grid(t, args.t_qmin, args.t_qmax, args.grid_size)

    # prepare DRNet bin edges based on global [qmin,qmax] range
    lo=float(t_grid[0]); hi=float(t_grid[-1])
    t_bins=np.linspace(lo, hi, args.drnet_bins+1).astype(np.float32)

    results=[]
    saved_pred={}

    for sd in seeds:
        set_seed(sd)
        # split
        n=len(X)
        idx=np.arange(n)
        rng=np.random.default_rng(sd)
        rng.shuffle(idx)
        n_te=int(args.test_size*n)
        te_idx=idx[:n_te]; tr_idx=idx[n_te:]

        Xtr,Xte=X[tr_idx],X[te_idx]
        Ttr,Tte=t[tr_idx],t[te_idx]
        Ytr,Yte=y[tr_idx],y[te_idx]
        p90=float(np.mean(Yte>=89.999))
        rmse_90=float(np.sqrt(np.mean((Yte-90.0)**2)))
        rmse_mean=float(np.sqrt(np.mean((Yte-Yte.mean())**2)))
        print(f"[seed {sd}] test stats: p90={p90:.3f} | RMSE(pred90)={rmse_90:.4f} | std={rmse_mean:.4f}")

        # standardize X (like your IHDP runs)
        sc=StandardScaler()
        Xtr=sc.fit_transform(Xtr)
        Xte=sc.transform(Xte)

        # train S-learner
        sle=SLearner(Xtr.shape[1], hidden=args.hidden, depth=args.depth, dropout=args.dropout, n_freq=args.n_freq).to(device)
        out_s=train_eval("slearner", sle, Xtr,Ttr,Ytr,Xte,Tte,Yte, t_grid=t_grid, device=device,
                         lr=args.lr, wd=args.wd, batch=args.batch, epochs=args.epochs, patience=args.patience)
        results.append({"seed":sd,"method":"S-learner MLP",**{k:out_s[k] for k in ["rmse","mae"]}})
        saved_pred.setdefault("S-learner MLP", []).append(out_s["mu_hat_grid"])

        # train VCNet-like
        vcn=VCNetLike(Xtr.shape[1], hidden=args.hidden, depth=args.depth, dropout=args.dropout, n_freq=args.n_freq).to(device)
        out_v=train_eval("vcnet", vcn, Xtr,Ttr,Ytr,Xte,Tte,Yte, t_grid=t_grid, device=device,
                         lr=args.lr, wd=args.wd, batch=args.batch, epochs=args.epochs, patience=args.patience)
        results.append({"seed":sd,"method":"VCNet-like",**{k:out_v[k] for k in ["rmse","mae"]}})
        saved_pred.setdefault("VCNet-like", []).append(out_v["mu_hat_grid"])

        # train DRNet-like
        drn=DRNetLike(Xtr.shape[1], K=args.drnet_bins, hidden=args.hidden, depth=args.depth, dropout=args.dropout, n_freq=args.n_freq).to(device)
        out_d=train_eval("drnet", drn, Xtr,Ttr,Ytr,Xte,Tte,Yte, t_grid=t_grid, device=device,
                         lr=args.lr, wd=args.wd, batch=args.batch, epochs=args.epochs, patience=args.patience,
                         K=args.drnet_bins, t_bins=t_bins)
        results.append({"seed":sd,"method":"DRNet-like",**{k:out_d[k] for k in ["rmse","mae"]}})
        saved_pred.setdefault("DRNet-like", []).append(out_d["mu_hat_grid"])

        print(f"[seed {sd}] done | RMSE: S={out_s['rmse']:.4f} VC={out_v['rmse']:.4f} DR={out_d['rmse']:.4f}")

    # aggregate & save
    df=pd.DataFrame(results)
    summ=df.groupby("method")[["rmse","mae"]].agg(["mean","std","count"]).reset_index()
    out_dir=os.path.join(args.out_dir, args.tag)
    df.to_csv(os.path.join(out_dir,"raw.csv"), index=False)
    summ.to_csv(os.path.join(out_dir,"summary.csv"), index=False)

    # save pred grids (mean over seeds)
    pred_out={}
    for k, lst in saved_pred.items():
        pred_out[k]=np.mean(np.stack(lst, axis=2), axis=2).astype(np.float32)  # (n_test_like, G) not aligned across seeds, so only for qualitative use
    np.savez_compressed(os.path.join(out_dir,"pred_grids.npz"), t_grid=t_grid, t_bins=t_bins, **{k.replace(" ","_"):v for k,v in pred_out.items()})

    # optional: evaluate ORCA pred (observed-point RMSE by interpolation) if provided
    if args.orca_pred and os.path.exists(args.orca_pred):
        z=np.load(args.orca_pred, allow_pickle=True)
        mu=z["mu_hat_grid"]  # expected (n_test, G, S) or (n_test,G)
        tg=z["t_grid"].astype(float)
        yte=z["y_test"].astype(float)
        tte=z["t_test"].astype(float)
        if mu.ndim==3:
            # per seed in file
            rmses=[]
            for i in range(mu.shape[2]):
                yhat=interp_grid_to_tobs(mu[:,:,i], tg, tte)
                rmses.append(rmse(yhat, yte))
            orca_rm=float(np.mean(rmses))
            orca_sd=float(np.std(rmses))
        else:
            yhat=interp_grid_to_tobs(mu, tg, tte)
            orca_rm=rmse(yhat, yte); orca_sd=0.0
        print(f"[ORCA pred] observed-point RMSE mean/std = {orca_rm:.6f} {orca_sd:.6f}")

    print("\n=== Summary (lower is better) ===")
    print(summ.sort_values(("rmse","mean")).to_string(index=False))

if __name__=="__main__":
    main()
