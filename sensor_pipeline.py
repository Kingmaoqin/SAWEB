# sensor_pipeline.py
# ------------------------------------------------------------
# Build a tabular dataset for survival analysis from sensor time-series files.
# Input:
#   1) A ZIP of sensor files: one file per subject/sample (CSV/Parquet).
#   2) A labels CSV with columns: file, duration, event  (file matches the basename in ZIP)
# Output:
#   pandas.DataFrame with columns: ['duration','event','sens_feat_...']
# Key points:
#   - Full-sequence features (no sliding window)
#   - Optional resampling to a target Hz if timestamp is present
#   - Robust per-channel stats + simple spectral features
# ------------------------------------------------------------
from __future__ import annotations

import io
import math
import os
import re
import tempfile
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

# Optional SciPy import (not strictly required; NumPy covers the basics)
# import scipy.signal as sps

NUMERIC_DTYPES = ("int8","int16","int32","int64","float16","float32","float64")

# ---------- I/O & utils ----------

def unzip_sensors_to_temp(uploaded_zip_file) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="sa_sensorzip_")
    with zipfile.ZipFile(uploaded_zip_file) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir

def scan_sensor_files(root: str, exts: Iterable[str] = (".csv", ".parquet", ".txt")):
    files = []
    for d, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                abspath = os.path.join(d, fn)
                rel = os.path.relpath(abspath, root)
                files.append((rel.replace("\\", "/"), abspath))
    files.sort()
    return files

def _read_one_table(path: str, time_cols=('timestamp','time','datetime','date','Time','Timestamp'), max_rows:int=0) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            import pyarrow  # noqa: F401
            df = pd.read_parquet(path)
        except Exception:
            # fallback to pandas engine if available
            df = pd.read_parquet(path)
    else:
        # CSV/TXT
        kw = dict(low_memory=False)
        if max_rows and max_rows > 0:
            kw["nrows"] = int(max_rows)
        df = pd.read_csv(path, **kw)

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # Try to automatically locate a timestamp column (prefer common names)
    ts_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in [t.lower() for t in time_cols]:
            ts_col = c
            break
    if ts_col is None:
        # Fallback: attempt to parse the first column as datetime
        c0 = df.columns[0]
        try:
            pd.to_datetime(df[c0])
            ts_col = c0
        except Exception:
            ts_col = None

    if ts_col is not None:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception:
            ts_col = None  # Parsing failed; treat as no timestamp

    return df, ts_col

# ---------- Feature engineering (full sequence) ----------

def _nanrobust(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]

def _basic_stats(x: np.ndarray) -> Dict[str, float]:
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
    # Simple zero-crossing rate
    s = np.sign(v - np.mean(v))
    out["zcr"]    = float(np.mean(s[1:] * s[:-1] < 0.0))
    # Skewness/kurtosis with NaN safety
    if v.size >= 3:
        m3 = np.mean((v - out["mean"])**3)
        m2 = np.mean((v - out["mean"])**2)
        out["skew"]  = float(m3 / (m2**1.5 + 1e-12))
        m4 = np.mean((v - out["mean"])**4)
        out["kurt"]  = float(m4 / (m2**2 + 1e-12))
    else:
        out["skew"] = 0.0
        out["kurt"] = 0.0
    # Signal energy (mean square)
    out["energy"] = float(np.mean(v**2))
    return out

def _spectral_feats(x: np.ndarray, fs: Optional[float]) -> Dict[str, float]:
    v = _nanrobust(x)
    out = {}
    if v.size < 8:
        return out
    X = np.fft.rfft(v - np.mean(v))
    P = np.abs(X)**2
    if P.sum() <= 0:
        return out
    Pn = P / P.sum()
    out["spec_entropy"] = float(-np.sum(Pn * np.log(Pn + 1e-12)))
    # Dominant frequency relative energy (unitless)
    k = int(np.argmax(P))
    out["dom_power"] = float(P[k] / (P.sum() + 1e-12))
    # Only emit dom_freq_hz when fs is known to avoid constant zero columns
    if fs and fs > 0:
        freqs = np.fft.rfftfreq(v.size, d=1.0/fs)
        out["dom_freq_hz"] = float(freqs[k])
    return out


def _maybe_resample(df: pd.DataFrame, ts_col: Optional[str], target_hz: float) -> Tuple[pd.DataFrame, Optional[float]]:
    """Resample to a regular grid when target_hz>0; otherwise estimate fs if possible."""
    if ts_col is None:
        return df, None
    df = df.sort_values(ts_col)
    if target_hz and target_hz > 0:
        # Resample to target_hz using mean + interpolation
        df = df.set_index(ts_col).resample(pd.Timedelta(seconds=1.0/target_hz)).mean().interpolate().reset_index()
        fs = float(target_hz)
    else:
        # Estimate sampling rate from the median timestamp gap
        dt = df[ts_col].diff().dt.total_seconds().to_numpy()
        dt = dt[np.isfinite(dt) & (dt>0)]
        fs = float(1.0/np.median(dt)) if dt.size>0 else None
    return df, fs

def _group_xyz_columns(cols: List[str]) -> List[Tuple[str,str,str]]:
    """Detect *_x, *_y, *_z channel triplets (e.g. accel axes)."""
    out = []
    s = set(cols)
    for c in cols:
        if c.endswith("_x"):
            base = c[:-2]
            cx, cy, cz = f"{base}_x", f"{base}_y", f"{base}_z"
            if (cx in s) and (cy in s) and (cz in s):
                out.append((cx,cy,cz))
    return out

def fullsequence_features(
    df: pd.DataFrame,
    *,
    ts_col: Optional[str],
    resample_hz: float = 0.0,
    max_rows: int = 0,
    tail_frac: float = 0.2
) -> Dict[str, float]:
    num_cols = [c for c in df.columns if str(df[c].dtype) in NUMERIC_DTYPES]
    for meta in ("unit", "cycle"):
        if meta in num_cols:
            num_cols.remove(meta)

    df2, fs = _maybe_resample(df, ts_col, target_hz=resample_hz)

    def _trend_feats(v: np.ndarray) -> Dict[str, float]:
        v = _nanrobust(v); n = v.size
        if n < 3: return {}
        t = np.arange(n, dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-12)
        slope, intercept = np.polyfit(t, v, 1)
        yhat = slope * t + intercept
        ss_res = np.sum((v - yhat) ** 2)
        ss_tot = np.sum((v - v.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return {"trend_slope": float(slope), "trend_r2": float(r2),
                "last_value": float(v[-1]), "delta_last_first": float(v[-1]-v[0])}

    def _tail_stats(v: np.ndarray, frac: float = 0.2) -> Dict[str, float]:
        v = _nanrobust(v); n = v.size
        if n < 4: return {}
        k = max(3, int(np.floor(n * frac)))
        tail = v[-k:]
        return {"tail_mean": float(np.mean(tail)),
                "tail_std": float(np.std(tail)),
                "tail_median": float(np.median(tail)),
                "tail_rms": float(np.sqrt(np.mean(tail**2)))}

    feats: Dict[str, float] = {}
    for c in num_cols:
        x = df2[c].to_numpy(dtype=np.float64, copy=False)
        for k, v in {**_basic_stats(x),
                     **_spectral_feats(x, fs),
                     **_trend_feats(x),
                     **_tail_stats(x, tail_frac)}.items():
            feats[f"{c}__{k}"] = v

    if ts_col is not None and len(df2) > 1:
        feats["seq_duration_sec"] = float((df2[ts_col].iloc[-1]-df2[ts_col].iloc[0]).total_seconds())
    if fs: feats["fs_hz_est"] = float(fs)
    return feats


    def _tail_stats(v: np.ndarray, frac: float = 0.2) -> Dict[str, float]:
        v = _nanrobust(v)
        n = v.size
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

    feats: Dict[str, float] = {}

    # Per-channel features: global stats + spectral + trend + tail windows
    for c in num_cols:
        x = df2[c].to_numpy(dtype=np.float64, copy=False)
        s1 = _basic_stats(x)
        s2 = _spectral_feats(x, fs)      # dom_freq_hz omitted when fs is None
        s3 = _trend_feats(x)
        s4 = _tail_stats(x, tail_frac)
        for k, v in {**s1, **s2, **s3, **s4}.items():
            feats[f"{c}__{k}"] = v

    # If a timestamp column exists, record sequence duration and fs estimate
    if ts_col is not None:
        dur_sec = (df2[ts_col].iloc[-1] - df2[ts_col].iloc[0]).total_seconds() if len(df2) > 1 else 0.0
        feats["seq_duration_sec"] = float(dur_sec)
    if fs:
        feats["fs_hz_est"] = float(fs)
    return feats


def sensors_to_dataframe(
    manifest_df: pd.DataFrame,
    sensor_root: str,
    *,
    file_col: str = "file",
    duration_col: str = "duration",
    event_col: str = "event",
    id_col: Optional[str] = "id",
    resample_hz: float = 0.0,
    max_rows_per_file: int = 0,
    time_cols=('timestamp','time','datetime','date','Time','Timestamp')
) -> pd.DataFrame:
    """Convert per-subject sensor files into a feature table aligned with duration/event labels."""
    for col in (file_col, duration_col, event_col):
        if col not in manifest_df.columns:
            raise KeyError(f"Manifest is missing column '{col}'. Required columns: file, duration, event.")
    rows = []
    dur_list = []
    evt_list = []
    id_list: Optional[List] = [] if (id_col and id_col in manifest_df.columns) else None

    available = scan_sensor_files(sensor_root)
    rel_lookup = {rel: abspath for rel, abspath in available}
    rel_lookup.update({rel.lstrip("./"): abspath for rel, abspath in available})
    basename_lookup: Dict[str, List[str]] = {}
    for rel, abspath in available:
        basename_lookup.setdefault(os.path.basename(rel), []).append(abspath)

    missing_paths: List[str] = []
    ambiguous_paths: List[str] = []

    for idx, r in manifest_df.iterrows():
        rel = str(r[file_col]).strip()
        rel_norm = rel.replace("\\", "/")
        abspath: Optional[str] = None
        if os.path.isabs(rel_norm) and os.path.exists(rel_norm):
            abspath = rel_norm
        else:
            abspath = rel_lookup.get(rel_norm)
            if abspath is None:
                abspath = rel_lookup.get(rel_norm.lstrip("/"))
            if abspath is None:
                candidates = basename_lookup.get(os.path.basename(rel_norm), [])
                if len(candidates) == 1:
                    abspath = candidates[0]
                elif len(candidates) > 1:
                    ambiguous_paths.append(rel)
                    continue
        if abspath is None or not os.path.exists(abspath):
            missing_paths.append(rel)
            continue
        try:
            df, ts_col2 = _read_one_table(abspath, time_cols=time_cols, max_rows=max_rows_per_file)
            feats = fullsequence_features(df, ts_col=ts_col2, resample_hz=resample_hz, max_rows=max_rows_per_file)
            rows.append(feats)
            # Collect scalars separately to avoid list-of-tuples ambiguity
            dur_list.append(float(pd.to_numeric(r[duration_col], errors="coerce")))
            evt_list.append(int(pd.to_numeric(r[event_col], errors="coerce")))
            if id_list is not None:
                id_list.append(r[id_col])
        except Exception as e:
            print(f"[sensor_pipeline] Skipped {rel}: {e}")
            continue

    if not rows:
        detail = []
        if missing_paths:
            missing_unique = sorted(set(missing_paths))
            preview = ", ".join(missing_unique[:10])
            if len(missing_unique) > 10:
                preview += "…"
            detail.append(f"Missing files: {preview}")
        if ambiguous_paths:
            ambiguous_unique = sorted(set(ambiguous_paths))
            preview = ", ".join(ambiguous_unique[:10])
            if len(ambiguous_unique) > 10:
                preview += "…"
            detail.append(f"Ambiguous matches (duplicate filenames in ZIP): {preview}")
        hint = "; ".join(detail)
        raise RuntimeError(
            "No sensor features could be extracted. Ensure the manifest 'file' column matches entries inside the ZIP archive."
            + (f" {hint}" if hint else "")
        )

    # Feature table
    feat_df = pd.DataFrame(rows)

    # Column filtering: drop high-NaN, zero-variance, and dom_freq columns when fs is unknown
    nan_ratio = feat_df.isna().mean()
    std = feat_df.std(ddof=0)
    keep_cols = (nan_ratio <= 0.2) & (std > 1e-12)
    feat_df = feat_df.loc[:, keep_cols]
    drop_domfreq = [c for c in feat_df.columns if c.endswith("__dom_freq_hz")]
    if drop_domfreq:
        feat_df = feat_df.drop(columns=drop_domfreq)

    feat_df = feat_df.reindex(sorted(feat_df.columns), axis=1).fillna(0.0)

    # Label table aligned with successfully processed samples
    de = pd.DataFrame({
        "duration": pd.Series(dur_list, dtype="float32"),
        "event":    pd.Series(evt_list, dtype="int32"),
    })
    if id_list is not None:
        de.insert(0, id_col, pd.Series(id_list))

    # Safety check
    if len(de) != len(feat_df):
        raise RuntimeError(
            f"Number of labels and extracted feature rows differ: labels={len(de)} features={len(feat_df)}. "
            "Some sensor files may be missing or failed to parse."
        )

    # Optional global z-score normalisation (can be moved to trainer later)
    DO_GLOBAL_Z = True
    if DO_GLOBAL_Z:
        mu = feat_df.mean(axis=0)
        sigma = feat_df.std(axis=0).replace(0.0, 1.0)
        feat_df = (feat_df - mu) / sigma

    out = pd.concat([de, feat_df.astype(np.float32)], axis=1)
    # Apply a consistent prefix
    out.columns = ["duration","event"] + [f"sens_feat_{c}" if not str(c).startswith("sens_feat_") else c
                                        for c in out.columns[2:]]
    return out



def build_manifest_from_sensors_zip(uploaded_zip_file) -> Tuple[pd.DataFrame, str]:
    """Extract the ZIP and prepare a manifest (file, duration, event) with empty labels."""
    root = unzip_sensors_to_temp(uploaded_zip_file)
    files = [rel for rel,_ in scan_sensor_files(root)]
    if not files:
        raise RuntimeError("No sensor files were found in the ZIP (supported: .csv/.parquet/.txt).")
    df = pd.DataFrame({"file": files, "duration": np.nan, "event": 0})
    return df, root

def manifest_template_csv_bytes(manifest_df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    manifest_df[["file","duration","event"]].to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
