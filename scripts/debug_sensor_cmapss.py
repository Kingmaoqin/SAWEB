# scripts/debug_sensor_cmapss.py
# 用法：
#   python scripts/debug_sensor_cmapss.py \
#     --zip ~/data/cmapss/sa_official_cens/FD002_censored.zip \
#     --labels ~/data/cmapss/sa_official_cens/FD002_censored_labels.csv \
#     --outdir ~/data/cmapss/debug_fd002

import argparse, os, zipfile, tempfile, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def nanrobust(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]

def basic_stats(x: np.ndarray) -> dict:
    v = nanrobust(x)
    if v.size == 0:
        return {"mean":np.nan,"std":np.nan,"min":np.nan,"p05":np.nan,"p25":np.nan,
                "median":np.nan,"p75":np.nan,"p95":np.nan,"max":np.nan,"iqr":np.nan,
                "rms":np.nan,"absmean":np.nan,"zcr":np.nan,"skew":np.nan,"kurt":np.nan,
                "energy":np.nan}
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

def spectral_feats(x: np.ndarray) -> dict:
    v = nanrobust(x)
    if v.size < 8:
        return {"spec_entropy":np.nan,"dom_freq_hz":0.0,"dom_power":np.nan}
    X = np.fft.rfft(v - np.mean(v))
    P = np.abs(X)**2
    if P.sum() <= 0:
        return {"spec_entropy":np.nan,"dom_freq_hz":0.0,"dom_power":np.nan}
    Pn = P / P.sum()
    ent = float(-np.sum(Pn * np.log(Pn + 1e-12)))
    k = int(np.argmax(P))
    return {"spec_entropy":ent, "dom_freq_hz":0.0, "dom_power":float(P[k]/(P.sum()+1e-12))}

def fullseq_feats(df: pd.DataFrame) -> dict:
    # 仅数值列
    num_cols = [c for c in df.columns if str(df[c].dtype) in NUMERIC_DTYPES]
    feats = {}
    for c in num_cols:
        x = df[c].to_numpy(dtype=np.float64, copy=False)
        s1 = basic_stats(x); s2 = spectral_feats(x)
        for k,v in {**s1, **s2}.items():
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
    df_lbl = pd.read_csv(lbl_path)
    df_lbl.columns = [c.strip().lower() for c in df_lbl.columns]
    assert {"file","duration","event"}.issubset(df_lbl.columns), "标签需含 file,duration,event"

    # 1) 对齐性检查
    miss_in_zip = sorted(set(df_lbl["file"]) - set(files_in_zip))
    miss_in_lbl = sorted(set(files_in_zip) - set(df_lbl["file"]))
    print(f"[CHECK] zip文件数={len(files_in_zip)}  标签数={len(df_lbl)}")
    print(f"[CHECK] 标签有但zip缺失={len(miss_in_zip)}  zip有但标签缺失={len(miss_in_lbl)}")
    if miss_in_zip: (outdir/"missing_in_zip.txt").write_text("\n".join(miss_in_zip))
    if miss_in_lbl: (outdir/"missing_in_labels.txt").write_text("\n".join(miss_in_lbl))

    # 2) 标签合法性
    df_lbl["duration"] = pd.to_numeric(df_lbl["duration"], errors="coerce")
    df_lbl["event"]    = pd.to_numeric(df_lbl["event"], errors="coerce")
    bad_dur = df_lbl["duration"].le(0).sum() + df_lbl["duration"].isna().sum()
    bad_evt = (~df_lbl["event"].isin([0,1])).sum()
    print(f"[CHECK] 非法 duration 行={bad_dur}  非法 event 行={bad_evt}")

    # 3) 逐文件快速体检
    per_file = []
    for f in df_lbl["file"]:
        p = tmp / f
        if not p.exists():
            per_file.append((f, -1, "MISSING", 0, 0))
            continue
        df = read_table(p)
        nrows, ncols = df.shape
        # 数值列占比、全NaN列数
        num_cols = [c for c in df.columns if str(df[c].dtype) in NUMERIC_DTYPES]
        allnan_cols = [c for c in num_cols if df[c].isna().all()]
        # 检查 cycle 是否从1递增
        cyc_ok = False
        if "cycle" in df.columns:
            cs = pd.to_numeric(df["cycle"], errors="coerce")
            cyc_ok = cs.is_monotonic_increasing and cs.dropna().min() >= 1
        per_file.append((f, nrows, "OK" if cyc_ok else "CYCLE?", len(num_cols), len(allnan_cols)))
    pf = pd.DataFrame(per_file, columns=["file","n_rows","cycle_ok","n_numeric_cols","n_allnan_numeric_cols"])
    pf.to_csv(outdir/"per_file_summary.csv", index=False)
    print(f"[CHECK] per_file_summary.csv 已写出；空/异常行：",
          int((pf["n_rows"]<=1).sum()), "  全NaN数值列>0的文件：", int((pf["n_allnan_numeric_cols"]>0).sum()))

    # 4) 复算整段特征（和前端一致），看NaN/常数列
    rows = []
    keep_de = []
    for f, dur, evt in df_lbl[["file","duration","event"]].itertuples(index=False):
        p = tmp / f
        if not p.exists(): continue
        df = read_table(p)
        feats = fullseq_feats(df)
        rows.append(feats)
        keep_de.append((dur, evt))
    if not rows:
        print("[ERROR] 一个样本都没能提取出特征。")
        return
    X = pd.DataFrame(rows)
    X = X.reindex(sorted(X.columns), axis=1)
    X.to_csv(outdir/"feature_table_raw.csv", index=False)

    # 5) 列级体检：NaN占比、方差、常数列
    nan_ratio = X.isna().mean().rename("nan_ratio")
    std = X.std(ddof=0).rename("std")
    nunique = X.nunique().rename("nunique")
    stats = pd.concat([nan_ratio, std, nunique], axis=1).sort_values(["nan_ratio","std"])
    stats.to_csv(outdir/"feature_stats.csv")
    const_cols = stats[(stats["nunique"]<=1) | (stats["std"]<=1e-12)].index.tolist()
    bad_cols   = stats[stats["nan_ratio"]>0.5].index.tolist()
    print(f"[CHECK] 特征列总数={X.shape[1]}  常数/零方差列={len(const_cols)}  高NaN列(>50%)={len(bad_cols)}")
    (outdir/"constant_cols.txt").write_text("\n".join(const_cols))
    (outdir/"high_nan_cols.txt").write_text("\n".join(bad_cols))

    # 6) 与 duration 的粗相关（Spearman），看看有没有信号
    df_de = pd.DataFrame(keep_de, columns=["duration","event"])
    X_filled = X.fillna(0.0)
    corr = X_filled.apply(lambda c: c.corr(df_de["duration"], method="spearman"))
    top = corr.abs().sort_values(ascending=False).head(20)
    corr[top.index].to_csv(outdir/"top20_corr_with_duration.csv")
    print("[CHECK] top20 与 duration 的Spearman相关 已写出。")

    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_constant_features": int(len(const_cols)),
        "n_high_nan_features": int(len(bad_cols)),
        "examples_constant": const_cols[:10],
        "examples_high_nan": bad_cols[:10]
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[DONE] 结果目录：", outdir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
