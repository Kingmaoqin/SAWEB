"""Routing and adapter utilities for TEXGISA backends.

This module provides a dual-path execution strategy:
1) Internal SAWEB implementation (models.mysa.run_mysa)
2) External texgisa-survival package backend

The returned payload is normalized to match the keys expected by the
existing Streamlit UI.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd



def _as_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _ensure_survival_frame(
    survival: Any,
    n_samples: int,
    n_bins_hint: int = 10,
) -> pd.DataFrame:
    """Convert package survival output into UI-compatible DataFrame.

    UI expects rows=time bins and columns=samples.
    """

    if isinstance(survival, pd.DataFrame):
        surv = survival.copy()
        return surv

    arr = np.asarray(survival, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Common conventions: (n_samples, n_times) or (n_times, n_samples).
    if arr.ndim == 2:
        if arr.shape[0] == n_samples and arr.shape[1] != n_samples:
            arr = arr.T
        elif arr.shape[1] == n_samples:
            pass
        elif arr.shape[0] < arr.shape[1]:
            # Prefer times x samples; transpose if first dim is likely samples.
            arr = arr.T
    else:
        arr = np.tile(np.linspace(1.0, 0.2, n_bins_hint).reshape(-1, 1), (1, n_samples))

    if arr.shape[1] != n_samples:
        # Final fallback to deterministic placeholder compatible with UI.
        arr = np.tile(np.linspace(1.0, 0.2, max(2, n_bins_hint)).reshape(-1, 1), (1, n_samples))

    cols = [f"sample_{i}" for i in range(n_samples)]
    idx = np.arange(1, arr.shape[0] + 1)
    return pd.DataFrame(arr, index=idx, columns=cols)


def _normalize_feature_importance(
    importance_result: Any,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Normalize package importance payload to current UI schema."""

    if isinstance(importance_result, pd.DataFrame):
        df = importance_result.copy()
        if "feature" not in df.columns:
            df.insert(0, "feature", list(feature_names)[: len(df)])
        if "importance" not in df.columns:
            numeric_cols = [c for c in df.columns if c != "feature"]
            if numeric_cols:
                df = df.rename(columns={numeric_cols[0]: "importance"})
            else:
                df["importance"] = 0.0
        return df

    if isinstance(importance_result, dict):
        raw_imp = importance_result.get("importance")
        raw_std = importance_result.get("std")
        raw_dir = importance_result.get("directional_mean")
    else:
        raw_imp, raw_std, raw_dir = importance_result, None, None

    imp = _as_float_array(raw_imp) if raw_imp is not None else np.zeros(len(feature_names), dtype=float)
    if imp.size != len(feature_names):
        imp = np.resize(imp, len(feature_names))

    out = {
        "feature": list(feature_names),
        "importance": imp.astype(float),
    }

    if raw_std is not None:
        std = _as_float_array(raw_std)
        if std.size != len(feature_names):
            std = np.resize(std, len(feature_names))
        out["std"] = std.astype(float)

    if raw_dir is not None:
        dm = _as_float_array(raw_dir)
        if dm.size != len(feature_names):
            dm = np.resize(dm, len(feature_names))
        out["directional_mean"] = dm.astype(float)

    fi_df = pd.DataFrame(out)
    fi_df = fi_df.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    return fi_df


def _backend_from_config(config: Dict[str, Any]) -> str:
    cfg_choice = str(config.get("texgisa_backend", "")).strip().lower()
    env_choice = os.getenv("TEXGISA_BACKEND", "").strip().lower()
    choice = cfg_choice or env_choice or "internal"
    if choice not in {"internal", "package", "auto"}:
        return "internal"
    return choice


def _run_package_backend(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run TEXGISA via external texgisa-survival package and normalize outputs."""

    texgisa_mod = importlib.import_module("texgisa_survival")
    TexGISa = getattr(texgisa_mod, "TexGISa")

    feature_cols = list(config.get("feature_cols") or [c for c in data.columns if c not in {"duration", "event"}])
    X = data[feature_cols].to_numpy(dtype=float)
    time = data["duration"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=int)

    model_kwargs = {
        "num_time_bins": int(config.get("num_intervals", 10)),
        "lambda_expert": float(config.get("lambda_expert", 0.0)),
        "ig_steps": int(config.get("ig_steps", 20)),
    }
    # Keep only kwargs accepted by package constructor at runtime.
    try:
        import inspect

        sig = inspect.signature(TexGISa.__init__)
        accepted = set(sig.parameters.keys())
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted}
    except Exception:
        pass

    model = TexGISa(**model_kwargs)

    # Best-effort expert-rule mapping for Phase 0/1.
    expert_cfg = config.get("expert_rules") or {}
    rules = expert_cfg.get("rules") if isinstance(expert_cfg, dict) else expert_cfg
    for rule in rules or []:
        feature = rule.get("feature")
        if feature not in feature_cols:
            continue
        sign = int(rule.get("sign", 1) or 1)
        weight = float(rule.get("weight", 1.0))
        threshold = rule.get("threshold", "mean")
        relation = rule.get("relation", ">")
        if hasattr(model, "add_expert_rule"):
            model.add_expert_rule(
                feature=feature,
                relation=relation,
                threshold=threshold,
                sign=sign,
                weight=weight,
            )

    fit_kwargs = {
        "epochs": int(config.get("epochs", 200)),
        "batch_size": int(config.get("batch_size", 128)),
        "lr": float(config.get("lr", 1e-3)),
        "verbose": 0,
    }

    try:
        import inspect

        fit_sig = inspect.signature(model.fit)
        accepted_fit = set(fit_sig.parameters.keys())
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted_fit}
    except Exception:
        pass

    model.fit(X, time, event, **fit_kwargs)

    risk_scores = _as_float_array(model.predict_risk(X)).astype(float)
    survival = model.predict_survival(X)
    surv_df = _ensure_survival_frame(survival, n_samples=X.shape[0], n_bins_hint=int(config.get("num_intervals", 10)))

    fi_raw = model.get_feature_importance(method="texgi") if hasattr(model, "get_feature_importance") else None
    fi_df = _normalize_feature_importance(fi_raw, feature_cols)

    return {
        "algo": "TEXGISA",
        "backend": "package",
        "C-index (Validation)": np.nan,
        "Num Bins": int(surv_df.shape[0]),
        "Val Samples": int(X.shape[0]),
        "Surv_Test": surv_df,
        "Feature Importance": fi_df,
        "risk_scores": risk_scores.tolist(),
        "Expert Rules Used": expert_cfg or {},
        "hazards": None,
        "cf_features": pd.DataFrame(X, columns=feature_cols),
        "cf_feature_stats": None,
        "cf_model_spec": None,
        "time_bin_stats": None,
        "time_unit_label": "time unit",
    }


def run_texgisa_dual_backend(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run TEXGISA using internal/package backend with optional auto fallback."""

    backend = _backend_from_config(config)

    if backend == "internal":
        from models.mysa import run_mysa

        result = run_mysa(data, config)
        result["backend"] = "internal"
        return result

    if backend == "package":
        return _run_package_backend(data, config)

    # auto mode: prefer package and gracefully fall back to internal.
    try:
        return _run_package_backend(data, config)
    except Exception as exc:
        from models.mysa import run_mysa

        result = run_mysa(data, config)
        result["backend"] = "internal"
        result["backend_fallback_reason"] = str(exc)
        return result
