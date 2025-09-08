# scripts/debug_sensor_cmapss_v2.py
# 用法：
#   python scripts/debug_sensor_cmapss_v2.py \
#     --zip ~/data/cmapss/sa_official_cens/FD002_censored.zip \
#     --labels ~/data/cmapss/sa_official_cens/FD002_censored_labels.csv \
#     --outdir ~/data/cmapss/debug_fd002_v2

import argparse, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_DTYPES = ("int8","int16","int32","int64","float16","float32","float64")

def unzip(zip_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(outdir)
    return outdir

def read_table(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _nanrobust(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]

def _basic_stats(x: np.ndarray) -> dict:
    v = _nanrobust(x)
    if v.size == 0:
        return {}
    out = {}
    out["mean"]   = float(np.mean(v))
    out["std"]    = float(np.std(v))
    out["min"]    = float(np.min(v))
    out["p05"]    = float(np.percentile(v, 5))
    out["p25"]    = float(np.percentile(v, 25))
    out["median"] = float(np.median(v))
    out["p75"]    = float(np.percentile(v, 75))
    out["p95"]    = float(np.percentile(v, 95))
    out["max"]    = float(np.max(v))
    out["iqr"]    = out["p75"] - out["p25"]
    out["rms"]    = float(np.sqrt(np.mean(v**2)))
    out["absmean"]= float(np.mean(np.abs(v)))
    s = np.sign(v - np.mean(v))
    out["zcr"]    = float(np.mean(s[1:] * s[:-1] < 0.0)) if v.size>1 else 0.0
    if v.size >= 3:
        m3 = np.mean((v - out["mean"])**3); m2 = np.mean((v - out["mean"])**2)
        out["skew"]  = float(m3 / (m2**1.5 + 1e-12))
        m4 = np.mean((v - out["mean"])**4)
        out["kurt"]  = float(m4 / (m2**2 + 1e-12))
    else:
        out["skew"] = 0.0; out["kurt"] = 0.0
    out["energy"] = float(np.mean(v**2))
    return out

def _spectral_feats_no_fs(x: np.ndarray) -> dict:
    # 与新版管线一致：无采样率 -> 只给谱熵和主频相对能量，不产出 dom_freq_hz
    v = _nanrobust(x)
    if v.size < 8:
        return {}
    X = np.fft.rfft(v - np.mean(v))
    P = np.abs(X)**2
    if P.sum() <= 0:
        return {}
    Pn = P / P.sum()
    ent = float(-np.sum(Pn * np.log(Pn + 1e-12)))
    k = int(np.argmax(P))
    return {"spec_entropy": ent, "dom_power": float(P[k] / (P.sum() + 1e-12))}

def _trend_feats(v: np.ndarray) -> dict:
    v = _nanrobust(v); n = v.size
    if n < 3:
        return {}
    t = np.arange(n, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)
    slope, intercept = np.polyfit(t, v, 1)
    yhat = slope * t + intercept
    ss_res = np.sum((v - yhat)**2)
    ss_tot = np.sum((v - v.mean())**2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return {
        "trend_slope": float(slope),
        "trend_r2": float(r2),
        "last_value": float(v[-1]),
        "delta_last_first": float(v[-1] - v[0]),
    }

def _tail_stats(v: np.ndarray, frac: float = 0.2) -> dict:
    v = _nanrobust(v); n = v.size
    if n < 4:
        return {}
    k = max(3, int(np.floor(n * frac)))
    tail = v[-k:]
    return {
        "tail_mean": float(np.mean(tail)),
        "tail_std": float(np.std(tail)),
        "tail_median": float(np.median(tail)),
        "tail_rms": float(np.sqrt(np.mean(tail ** 2))),
    }

def extract_fullseq_features(df: pd.DataFrame, tail_frac: float = 0.2) -> dict:
    # 与新版管线一致：只对真实传感/工况列做特征，排除 unit/cycle
    num_cols = [c for c in df.columns if str(df[c].dtype) in NUMERIC_DTYPES]
    for meta in ("unit","cycle"):
        if meta in num_cols:
            num_cols.remove(meta)
    feats = {}
    for c in num_cols:
        x = df[c].to_numpy(dtype=np.float64, copy=False)
        s1 = _basic_stats(x)
        s2 = _spectral_feats_no_fs(x)
        s3 = _trend_feats(x)
        s4 = _tail_stats(x, tail_frac)
        for k, v in {**s1, **s2, **s3, **s4}.items():
            feats[f"{c}__{k}"] = v
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    zip_path = Path(args.zip).expanduser()
    lbl_path = Path(args.labels).expanduser()
    outdir = Path(args.outdir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)

    tmp = unzip(zip_path, outdir / "unzipped")
    files_in_zip = sorted([str(p.relative_to(tmp)) for p in tmp.rglob("*.csv")])
    lbl = pd.read_csv(lbl_path)
    lbl.columns = [c.strip().lower() for c in lbl.columns]

    # 对齐性
    miss_in_zip = sorted(set(lbl["file"]) - set(files_in_zip))
    miss_in_lbl = sorted(set(files_in_zip) - set(lbl["file"]))
    print(f"[CHECK] zip={len(files_in_zip)}  labels={len(lbl)}  miss_zip={len(miss_in_zip)}  miss_lbl={len(miss_in_lbl)}")

    # 抽特征（新版逻辑）
    rows, de = [], []
    for f, dur, evt in lbl[["file","duration","event"]].itertuples(index=False):
        p = tmp / f
        if not p.exists(): continue
        df = read_table(p)
        feats = extract_fullseq_features(df, tail_frac=0.2)
        rows.append(feats)
        de.append((dur, evt))
    X = pd.DataFrame(rows)

    # 列清理（与新版管线一致）
    nan_ratio = X.isna().mean()
    std = X.std(ddof=0)
    keep = (nan_ratio <= 0.2) & (std > 1e-12)
    X = X.loc[:, keep].reindex(sorted(X.columns), axis=1).fillna(0.0)

    # 统计输出
    const_cols = X.columns[(X.std(ddof=0) <= 1e-12)]
    domfreq_cols = [c for c in X.columns if c.endswith("__dom_freq_hz")]
    unit_cols = [c for c in X.columns if c.startswith("unit__")]
    cycle_cols = [c for c in X.columns if c.startswith("cycle__")]

    print(f"[STATS] n_samples={X.shape[0]}  n_features={X.shape[1]}")
    print(f"[STATS] constant_cols={len(const_cols)}  dom_freq_cols={len(domfreq_cols)}  unit_cols={len(unit_cols)}  cycle_cols={len(cycle_cols)}")

    X.to_csv(outdir/"feature_table_afterfix.csv", index=False)

    # 与 duration 的相关（粗看是否有信号）
    de_df = pd.DataFrame(de, columns=["duration","event"])
    corr = X.apply(lambda c: c.corr(de_df["duration"], method="spearman"))
    corr.abs().sort_values(ascending=False).head(20).to_csv(outdir/"top20_corr_with_duration_afterfix.csv")

    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_constant_features": int(len(const_cols)),
        "n_dom_freq_cols": int(len(domfreq_cols)),
        "n_unit_cols": int(len(unit_cols)),
        "n_cycle_cols": int(len(cycle_cols)),
    }
    (outdir/"summary_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[DONE]", json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
