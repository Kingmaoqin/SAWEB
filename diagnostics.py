
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data import make_intervals, build_supervision, infer_feature_cols
from model import MultiTaskModel
from metrics import cindex_fast


def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def predict_hazards(model, X, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 1024):
            xb = torch.from_numpy(X[i:i+1024]).to(device)
            hb = model(xb).cpu().numpy()
            outs.append(hb)
    return np.concatenate(outs, axis=0)  # [N, T]


def brier_score_discrete(hazards, intervals, events):
    """
    Approximate Brier score per bin k using:
    target y_k = 1 if event occurred by k, else 0; predict p_k = 1 - S_k
    S_k = prod_{j<=k} (1 - h_j). We ignore IPCW for simplicity.
    Returns array [T] of mean squared error over all samples.
    """
    N, T = hazards.shape
    one_minus_h = 1.0 - hazards
    S = np.cumprod(one_minus_h, axis=1)
    P = 1.0 - S  # cumulative event prob by k
    bs = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int64)

    for i in range(N):
        k = int(intervals[i])
        e = int(events[i])
        if k <= 0:
            continue
        for j in range(T):
            y = 1 if (e == 1 and (j + 1) >= k) else 0
            bs[j] += (P[i, j] - y) ** 2
            cnt[j] += 1
    bs = np.divide(bs, np.maximum(cnt, 1))
    return bs


def calibration_by_quantile(hazards, intervals, events, k_idx, n_bins=10):
    """
    Decile calibration for cumulative event prob by bin k_idx (0-based).
    Returns (pred_mean per group, observed_event_rate per group, group sizes).
    """
    one_minus_h = 1.0 - hazards
    S = np.cumprod(one_minus_h, axis=1)
    Pk = 1.0 - S[:, k_idx]
    order = np.argsort(Pk)
    groups = np.array_split(order, n_bins)
    xs, ys, ns = [], [], []
    for g in groups:
        if len(g) == 0:
            continue
        pred_mean = Pk[g].mean()
        obs = ((events[g] == 1) & ((intervals[g] - 1) <= k_idx)).astype(np.float32).mean()
        xs.append(pred_mean)
        ys.append(obs)
        ns.append(len(g))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--duration_col", type=str, required=True)
    ap.add_argument("--event_col", type=str, required=True)
    ap.add_argument("--feature_cols", type=str, default="")
    ap.add_argument("--n_bins", type=int, default=30)
    ap.add_argument("--model_path", type=str, default="model.pt")
    ap.add_argument("--out_dir", type=str, default="diagnostics_out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSV (robust to case/space in header)
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()
    name_map = {c.lower(): c for c in df.columns}

    def resolve(colname):
        key = colname.strip().lower()
        if key in name_map:
            return name_map[key]
        raise KeyError(f"Column '{colname}' not found. Available: {list(df.columns)}")

    args.duration_col = resolve(args.duration_col)
    args.event_col = resolve(args.event_col)

    # Discretize durations
    df = make_intervals(
        df,
        duration_col=args.duration_col,
        event_col=args.event_col,
        n_bins=args.n_bins,
        method="quantile",
    )
    num_bins = int(df["interval_number"].max())

    # Feature columns
    if args.feature_cols.strip():
        feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        feat_cols = infer_feature_cols(df, exclude=[args.duration_col, args.event_col])

    durations = df[args.duration_col].to_numpy().astype(np.float32)
    events = df[args.event_col].to_numpy().astype(np.int32)
    intervals = df["interval_number"].to_numpy().astype(np.int32)
    X_all = df[feat_cols].to_numpy().astype(np.float32)

    # Standardize
    mu, sd = zscore_fit(X_all)
    X_all = zscore_apply(X_all, mu, sd)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskModel(input_dim=X_all.shape[1], num_bins=num_bins)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Predictions
    hazards = predict_hazards(model, X_all, device)  # [N, T]
    risks = hazards.sum(axis=1)  # [N]

    # Metrics
    cidx = cindex_fast(torch.tensor(durations), torch.tensor(events), torch.tensor(risks)).item()
    brier = brier_score_discrete(hazards, intervals, events)

    # Calibration at median bin
    k_idx = int(num_bins // 2) - 1
    xs, ys, ns = calibration_by_quantile(hazards, intervals, events, k_idx, n_bins=10)

    # Mean hazard over time
    mean_h = hazards.mean(axis=0)

    # Save numeric report
    report = {
        "num_samples": int(X_all.shape[0]),
        "num_features": int(X_all.shape[1]),
        "num_bins": int(num_bins),
        "cindex_overall": float(cidx),
        "brier_mean": float(brier.mean() if len(brier) > 0 else float("nan")),
        "calibration_bin": int(k_idx + 1),
        "calibration_points": [
            {"pred_mean": float(px), "obs_rate": float(oy), "n": int(n)}
            for px, oy, n in zip(xs, ys, ns)
        ],
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        import json
        json.dump(report, f, indent=2)

    # === PLOTS ===
    # 1) mean hazard per bin
    plt.figure()
    plt.plot(np.arange(1, num_bins + 1), mean_h)
    plt.xlabel("Time bin")
    plt.ylabel("Mean hazard")
    plt.title("Mean hazard over time")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mean_hazard_over_time.png"))
    plt.close()

    # 2) risk distribution
    plt.figure()
    plt.hist(risks, bins=30)
    plt.xlabel("Risk score (sum of hazards)")
    plt.ylabel("Count")
    plt.title("Risk distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "risk_distribution.png"))
    plt.close()

    # 3) sample survival curves
    one_minus_h = 1.0 - hazards
    S = np.cumprod(one_minus_h, axis=1)
    sel = np.random.choice(hazards.shape[0], size=min(12, hazards.shape[0]), replace=False)
    plt.figure()
    for i in sel:
        plt.plot(np.arange(1, num_bins + 1), S[i])
    plt.xlabel("Time bin")
    plt.ylabel("Survival probability S(t)")
    plt.title("Sample survival curves")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sample_survival_curves.png"))
    plt.close()

    # 4) Brier score over time
    plt.figure()
    plt.plot(np.arange(1, num_bins + 1), brier)
    plt.xlabel("Time bin")
    plt.ylabel("Brier score")
    plt.title("Brier score over time")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "brier_over_time.png"))
    plt.close()

    # 5) Calibration scatter
    plt.figure()
    plt.scatter(xs, ys)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Predicted event prob by bin k")
    plt.ylabel("Observed event rate by bin k (deciles)")
    plt.title(f"Calibration at bin k={k_idx + 1}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "calibration_scatter.png"))
    plt.close()

    # Per-bin stats CSV
    stats_df = pd.DataFrame({
        "bin": np.arange(1, num_bins + 1),
        "mean_hazard": mean_h,
        "brier": brier,
    })
    stats_df.to_csv(os.path.join(args.out_dir, "per_bin_stats.csv"), index=False)

    print(f"[OK] Diagnostics done. C-index={cidx:.4f}. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
