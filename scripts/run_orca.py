import argparse
import json
import numpy as np

from orca import ORCAConfig, load_npz_dataset, load_csv_dataset, fit_orca
from orca.utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True,
                    choices=["contT_contY","contT_binY","binT_contY","binT_binY"])
    ap.add_argument("--data_npz", type=str, default=None, help="NPZ with keys X_train,T_train,Y_train,X_test (+ optional truth).")
    ap.add_argument("--data_csv", type=str, default=None, help="CSV with feature cols + t,y (and optional split).")
    ap.add_argument("--out_dir", type=str, default="./runs/orca_demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    # quick knobs
    ap.add_argument("--nuisance", type=str, default="rf", choices=["rf","ridge","gbdt","mlp"])
    ap.add_argument("--tfeat", type=str, default="fourier", choices=["direct","fourier","mlp"])
    ap.add_argument("--crossfit_folds", type=int, default=1)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    cfg = ORCAConfig(
        task=args.task,
        nuisance=args.nuisance,
        tfeat=args.tfeat,
        crossfit_folds=args.crossfit_folds,
    )

    if args.data_npz is not None:
        ds = load_npz_dataset(args.data_npz, task=args.task)
    elif args.data_csv is not None:
        ds = load_csv_dataset(args.data_csv, task=args.task, t_col="t", y_col="y", split_col="split")
    else:
        raise ValueError("Provide --data_npz or --data_csv")

    model, metrics = fit_orca(ds, cfg, seed=args.seed, device=args.device)

    # save metrics
    with open(f"{args.out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "metrics": metrics, "cfg": cfg.__dict__}, f, indent=2)

    # save a lightweight prediction artifact
    t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.t_grid_size).astype(np.float32)
    if args.task.startswith("contT_"):
        pred = model.predict_contT_grid(ds.X_test, t_grid)
        np.save(f"{args.out_dir}/pred_grid.npy", pred)
        np.save(f"{args.out_dir}/t_grid.npy", t_grid)
    else:
        mu0, mu1 = model.predict_binT(ds.X_test)
        np.save(f"{args.out_dir}/mu0.npy", mu0)
        np.save(f"{args.out_dir}/mu1.npy", mu1)

    print("Done. metrics:", metrics)
    print("Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
