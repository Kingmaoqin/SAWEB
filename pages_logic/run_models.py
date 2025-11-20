
# run_models.py — MySA-integrated Workbench (drop-in replacement)
# This page supports: CoxTime, DeepSurv, DeepHit, and MySA (TEXGISA).
# - Adds two loss-weight sliders (lambda_smooth, lambda_expert)
# - Adds an editable Expert Rules table (relation/sign/min_mag/weight) and an "Important set" selector
# - Shows TEXGISA Feature Importance (Top-K + expand to all), and allows downloading time-dependent attributions.

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Set

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as pq

from models import coxtime, deepsurv, deephit
from models.mysa import run_mysa as run_texgisa
from utils.identifiers import canonicalize_series
from html import escape

_TOOLTIP_STYLE = """
<style>
.help-tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 1px solid #7b8794;
    color: #7b8794;
    font-weight: 700;
    font-size: 0.72rem;
    margin-left: 0.35rem;
    cursor: help;
    background: #ffffff10;
}

.help-tooltip:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.45);
}

.help-tooltip:focus-visible {
    outline: none;
}

.help-tooltip::after,
.help-tooltip::before {
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s ease-in-out;
}

.help-tooltip:hover::after,
.help-tooltip:focus::after,
.help-tooltip:hover::before,
.help-tooltip:focus::before {
    opacity: 1;
}

.help-tooltip::after {
    content: attr(data-tip);
    position: absolute;
    min-width: 200px;
    max-width: 320px;
    background: #1f2933;
    color: #f5f7fa;
    padding: 0.6rem 0.75rem;
    border-radius: 0.5rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.22);
    top: 125%;
    right: 0;
    white-space: pre-wrap;
    line-height: 1.4;
    z-index: 9999;
}

.help-tooltip::before {
    content: "";
    position: absolute;
    top: 108%;
    right: 8px;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-bottom: 7px solid #1f2933;
}

[data-testid="column"] > div {
    overflow: visible !important;
}

.stColumn {
    overflow: visible !important;
}
</style>
"""


def _render_help_tooltip(help_text: str, key: str) -> None:
    """Render a reusable ❔ tooltip with consistent styling."""

    if not help_text:
        return None

    flag_key = "_help_tooltip_css"
    if not st.session_state.get(flag_key):
        st.markdown(_TOOLTIP_STYLE, unsafe_allow_html=True)
        st.session_state[flag_key] = True

    safe_tip = escape(help_text).replace("\n", "<br/>")
    st.markdown(
        f"<span class='help-tooltip' data-tip='{safe_tip}' tabindex='0' id='{escape(key)}'>❔</span>",
        unsafe_allow_html=True,
    )
    return None
def _md_explain(text: str, size: str = "1.12rem", line_height: float = 1.6):
    """Render explanatory text using a larger font size for readability."""
    st.markdown(
        f"<div style='font-size:{size}; line-height:{line_height}; margin:0.25rem 0 1rem'>{text}</div>",
        unsafe_allow_html=True,
    )
def _explain_plot(kind: str, **kwargs) -> str:
    """
    Return an English paragraph interpreting the figure.
    kind ∈ {"surv_traj", "km_overall", "km_groups"}
    """
    import numpy as np

    if kind == "surv_traj":
        cols = kwargs.get("cols", [])
        last_vals = np.asarray(kwargs.get("last_vals", []), dtype=float)
        n = len(cols)
        if last_vals.size:
            avg_last = float(np.nanmean(last_vals))
            smin, smax = float(np.nanmin(last_vals)), float(np.nanmax(last_vals))
            spread = smax - smin
        else:
            avg_last, smin, smax, spread = float("nan"), float("nan"), float("nan"), 0.0

        return (
            f"This figure shows model-predicted survival trajectories for {n} sampled individuals. "
            "Each step down corresponds to a predicted event at that time bin; flatter lines indicate lower hazard. "
            f"On average, the end-of-horizon survival is around {avg_last:.2f} "
            f"(range {smin:.2f}–{smax:.2f}). "
            "Large separation or crossings between curves suggest heterogeneity in patient risk, "
            "while near-horizontal curves close to 1.0 imply few expected events within follow-up."
        )

    if kind == "km_overall":
        t = np.asarray(kwargs["t"], dtype=float)
        S = np.asarray(kwargs["S"], dtype=float)
        tail = float(S[-1]) if S.size else float("nan")

        # median survival if curve crosses 0.5
        median_txt = "The median survival is not reached within follow-up."
        if S.size and np.min(S) <= 0.5:
            # interpolate t where S == 0.5
            med = float(np.interp(0.5, S[::-1], t[::-1]))
            median_txt = f"The estimated median survival time is about {med:.1f}."

        return (
            "The Kaplan–Meier curve summarizes the observed survival probability over time. "
            "Step-downs occur at event times, and long flat segments reflect periods with censoring or no events. "
            f"At the last observed event time the survival probability is approximately {tail:.2f}. "
            f"{median_txt} "
            "A long flat tail typically indicates substantial censoring late in follow-up."
        )

    if kind == "km_groups":
        S_end = np.asarray(kwargs.get("S_end", []), dtype=float)  # survival at horizon for each group
        if S_end.size:
            order = np.argsort(S_end)  # low S_end -> high risk
            worst, best = int(order[0]) + 1, int(order[-1]) + 1
            sep = float(np.nanmax(S_end) - np.nanmin(S_end))
        else:
            worst = best = None
            sep = 0.0

        return (
            "Kaplan–Meier curves by risk groups illustrate how well the model separates patients. "
            "Lower curves indicate higher risk. "
            f"The spread between groups at the horizon is about {sep:.2f} in survival probability"
            + (f" (worst ≈ group {worst}, best ≈ group {best}). " if worst and best else ". ")
            + "Clear vertical separation suggests good discrimination; overlapping curves imply limited stratification."
        )

    if kind == "fi_topk":
        # df: top-k rows with columns ['feature','importance', ('directional_mean'?)]
        df = kwargs.get("df")
        import numpy as np, pandas as pd
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return (
                "This bar chart ranks features by TEXGISA feature importance "
                "(Time-dependent EXtreme Gradient Integration, TEXGISA). Larger bars indicate stronger contributions to the predicted risk over time."
            )

        k = int(len(df))
        top_feat = str(df.iloc[0]["feature"])
        top_imp = float(df.iloc[0]["importance"])
        avg_imp = float(np.nanmean(df["importance"].values))

        has_dir = "directional_mean" in df.columns
        pos = neg = 0
        if has_dir:
            s = df["directional_mean"].astype(float).fillna(0.0)
            pos = int((s > 0).sum())
            neg = int((s < 0).sum())

        msg = (
            f"This bar chart displays the top-{k} features ranked by TEXGISA feature importance (TEXGISA). "
            f"The most influential feature is {top_feat} (importance {top_imp:.4f}). "
            f"On average, the importance across the shown features is {avg_imp:.4f}. "
        )
        if has_dir:
            # Match the chart colour scheme: positive = blue (#60a5fa), negative = red (#f87171)
            msg += (
                "Bar colors encode direction: blue indicates a positive association with hazard "
                "(higher hazard → lower survival), and red indicates a negative association "
                "(lower hazard → higher survival) in the local attributions. "
                f"Among the top-{k} features, {pos} trend positive and {neg} trend negative. "
            )
        else:
            msg += "All bars share the same color because directional information is not available. "
        msg += (
            "Use this ranking to identify drivers of risk; consider domain knowledge to validate whether the signs "
            "and magnitudes align with plausible clinical effects."
        )
        return msg



    return ""


def _qhelp_md(key: str) -> str:
    """Maintain the help text (Markdown) for each parameter centrally. Add or modify entries as needed."""
    HELP = {
        # Data Mapping
        "time_col": "Name of the column that stores follow-up or survival time. The column must be numeric but the time units are flexible (days, months, years, and so on).",
        "event_col": "Binary event indicator column. Use 1 when the event occurs (for example death or recurrence) and 0 when the observation is censored.",
        "features": "Feature columns used for modeling. Do not include the time or event columns.",
        "positive_label": "Event-column label that should be mapped to 1. All other values are mapped to 0 for censored observations.",

        # Algorithm Selection & Common Training Parameters
        "algo": "Select the training algorithm. TEXGISA is the only option that supports end-to-end multimodal training on tabular data plus raw images or sensors while generating TEXGISA explanations.",
        "batch_size": "Number of samples processed per optimisation step. Larger batch sizes yield smoother gradients but increase GPU memory usage.",
        "epochs": "Total number of training epochs. Begin with 50–150 epochs to monitor convergence before scheduling longer runs.",
        "lr": "Learning rate for the optimiser. High values may diverge, whereas low values slow convergence. A starting range between 1e-3 and 1e-2 works well for most runs.",
        "val_split": "Fraction of the training data reserved for validation so early stopping and best-epoch selection can operate reliably.",

        # DeepHit
        "num_intervals": "Number of discrete time intervals used by DeepHit and other discrete-time models. Too many intervals create sparsity; too few limit temporal resolution.",

        # MySA Regularisation & Priors
        "lambda_expert": "Weight applied to the expert prior penalty (λ_expert). Higher values enforce the curated important feature set more strongly but may reduce predictive accuracy.",
        "lambda_smooth": "Temporal smoothness weight (λ_smooth). Encourages TEXGISA importance to vary smoothly over time but can hide sharp dynamics if set too high.",
        "important_features": "Expert-defined important feature set that λ_expert should emphasise during training.",
        "fast_mode": "Toggle for acceleration mode that uses an approximate TEXGISA pipeline for quick previews. Disable it for production-quality training.",
        "ig_steps": "Number of integration steps used when computing TEXGISA. Increasing the value improves accuracy at the cost of additional runtime.",
        "texgisa_constraints": "Editor for TEXGISA magnitude floors. Each row specifies a minimum TEXGISA importance enforced when λ_expert is greater than zero. Directional limits are not available yet.",
        "algo_panel": "Overview of algorithm-level controls shared across CoxTime, DeepSurv, DeepHit, and TEXGISA. Adjust these first before diving into model-specific priors.",
        "texgisa_regularizers": "TEXGISA introduces temporal smoothness (λ_smooth) and expert prior (λ_expert) penalties. Tune them here to balance interpretability, stability, and predictive performance.",
        "texgisa_guidance": "Configure expert-driven guidance for TEXGISA: highlight must-keep features and enforce minimum importance thresholds or penalties to align with domain knowledge.",
        "texgisa_advanced": "Fine-tune the TEXGISA generator and attribution pipeline. These knobs control integration steps, latent dimensions, and sampling budgets that affect explanation fidelity and runtime.",

        # Generator / TEXGISA Advanced Parameters
        "latent_dim": "Dimension of the generator latent noise vector. Larger dimensions capture more variation but usually require more data.",
        "extreme_dim": "Dimension of the extreme encoding vector that models high-risk trajectories.",
        "gen_epochs": "Number of generator training epochs (used only when TEXGISA is enabled).",
        "gen_batch": "Batch size used while training the generator.",
        "gen_lr": "Learning rate for the generator optimiser.",
        "gen_alpha_dist": "Distribution-distance regularisation weight α. Higher values keep generated samples close to the reference distribution.",
        "ig_batch_samples": "Number of samples B' drawn per TEXGISA batch. Use higher values for smoother gradients if runtime allows.",
        "ig_time_subsample": "Number of time steps T' sampled per TEXGISA pass to control integration speed.",
    }
    return HELP.get(key, "No description available (add a description for this key in _qhelp_md).")


def field_with_help(control_fn, label, help_key: str, *args, **kwargs):
    """Use Streamlit's built-in help= to match uploader style."""
    help_msg = _qhelp_md(help_key)
    return control_fn(label, *args, help=help_msg, **kwargs)



def uploader_with_help(label: str, *, key: str, help_text: str, **kwargs):
    """Streamlit ``file_uploader`` with a built-in help tooltip."""
    return st.file_uploader(label, key=key, help=help_text, **kwargs)


def _preview_dataframe(df: Optional[pd.DataFrame], *, max_rows: int = 10) -> None:
    """Display up to ``max_rows`` rows with consistent styling (handles wide tables gracefully)."""
    if df is None:
        return
    rows = min(len(df), max_rows)
    st.dataframe(df.head(rows), use_container_width=True)


def _preview_dataframe(df: Optional[pd.DataFrame], *, max_rows: int = 10) -> None:
    """Display up to ``max_rows`` rows with consistent styling (handles wide tables gracefully)."""
    if df is None:
        return
    rows = min(len(df), max_rows)
    st.dataframe(df.head(rows), use_container_width=True)


CHANNEL_HELP_TEXT: Dict[str, str] = {
    "tabular_csv": (
        "Upload a tabular survival dataset (CSV) with one row per subject, including `duration`/`event` columns. "
        "All algorithms can train on this table. Add a stable identifier column if you plan to align extra modalities."
    ),
    "img_zip_simple": (
        "Compressed folder (.zip) containing one image per subject. The wizard will extract features and build a table. "
        "Only TEXGISA can later train end-to-end on raw images; other algorithms consume the generated tabular features."
    ),
    "img_labels_csv": (
        "CSV template for image labels. Provide columns `image`, `duration`, and `event` so survival metrics can be derived."
    ),
    "sensor_zip": (
        "Compressed folder (.zip) with sensor files (CSV/Parquet) per subject. Ensure filenames map to individuals consistently. "
        "End-to-end multimodal training requires TEXGISA; other algorithms will use any derived feature table instead."
    ),
    "sensor_labels_csv": (
        "CSV with `file`, `duration`, and `event` columns describing outcomes for each sensor recording."
    ),
    "mm_tabular": (
        "Processed tabular modality aligned by ID. Works for every algorithm and acts as the anchor when merging other modalities."
    ),
    "mm_image": (
        "Image-level features or metadata keyed by the same identifier as the tabular table. TEXGISA is required for end-to-end multimodal learning."
    ),
    "mm_sensor": (
        "Sensor feature table keyed by the shared identifier. Non-TEXGISA algorithms will treat these as pre-merged columns."
    ),
    "mm_raw_tabular": (
        "Primary tabular CSV used to align raw assets. Must contain the shared identifier, duration, and event columns."
    ),
    "mm_raw_img_zip": (
        "Zip archive with raw image files referenced by the image manifest. Required for TEXGISA end-to-end multimodal runs."
    ),
    "mm_raw_img_manifest": (
        "CSV manifest that lists each image filename alongside the shared identifier so TEXGISA can stream the raw assets."
    ),
    "mm_raw_sensor_zip": (
        "Zip archive with raw sensor files. TEXGISA consumes these during joint multimodal optimisation."
    ),
    "mm_raw_sensor_manifest": (
        "CSV manifest describing sensor files (`file`, shared ID, optional metadata) for TEXGISA's multimodal loader."
    ),
}


def _extract_fi_df(results: dict) -> pd.DataFrame | None:
    """
    Try to extract a DataFrame with columns ['feature','importance',('directional_mean'?)] from results.
    Accepts many shapes: DataFrame / dict-of-lists / list-of-dicts / nested {'table': DataFrame} / synonyms.
    """

    if not isinstance(results, dict):
        return None

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        df = df.copy()

        # Normalise column names by casting them to lowercase strings for matching.
        cols_lower = [str(c).lower() for c in df.columns]

        # If the feature column lives in the index (common for multi-index inputs), reset the index first.
        has_index_name = (getattr(df.index, "name", None) is not None) or (
            hasattr(df.index, "names") and any(n is not None for n in (df.index.names or []))
        )
        if "feature" not in cols_lower and has_index_name:
            df = df.reset_index()
            cols_lower = [str(c).lower() for c in df.columns]

        # Map lowercase string column names back to their originals.
        colmap = {str(c).lower(): c for c in df.columns}

        # 1) Identify the feature column.
        feat_col = None
        for cand in ("feature", "index", "name", "variable", "feat", "feature_name"):
            if cand in colmap:
                feat_col = colmap[cand]
                break
        if feat_col is None:
            # Fallback: prefer the first object column when no explicit feature column exists.
            obj_cols = [c for c in df.columns if df[c].dtype == "O"]
            if obj_cols:
                feat_col = obj_cols[0]
            else:
                # As a last resort, use the first column when it looks categorical/string-like.
                if len(df.columns) >= 1 and df[df.columns[0]].dtype == "O":
                    feat_col = df.columns[0]
                else:
                    return None

        # 2) Locate the importance column, accounting for common aliases.
        imp_col = None
        for cand in ("importance", "score", "weight", "texgi", "attr", "phi", "value"):
            if cand in colmap:
                imp_col = colmap[cand]
                break
        if imp_col is None:
            # Fallback: choose the first numeric column that is not the feature column.
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != feat_col]
            if num_cols:
                imp_col = num_cols[0]
            else:
                return None

        # 3) Optional directional column.
        dir_col = None
        for cand in ("directional_mean", "direction", "signed_mean", "signed"):
            if cand in colmap:
                dir_col = colmap[cand]
                break

        # Rename the detected columns to the canonical schema.
        rename_map = {feat_col: "feature", imp_col: "importance"}
        if dir_col:
            rename_map[dir_col] = "directional_mean"
        df = df.rename(columns=rename_map)

        # Keep the canonical columns and normalise dtypes.
        keep = ["feature", "importance"] + (["directional_mean"] if "directional_mean" in df.columns else [])
        df = df[keep]
        df["feature"] = df["feature"].astype(str)
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        if "directional_mean" in df.columns:
            df["directional_mean"] = pd.to_numeric(df["directional_mean"], errors="coerce")
        df = df.dropna(subset=["importance"])
        return df

    # 1) Look for well-known keys first.
    candidates = [
        "fi_table", "texgi_importance", "fi_df", "feature_importance", "texgi_fi",
        "texgi_importance_df", "TEXGI", "fi", "fi_table_full", "importance_table",
        "texgi_ranked", "texgi_topk",
    ]
    for k in candidates:
        if k in results:
            v = results[k]
            if isinstance(v, pd.DataFrame):
                df = _normalize_df(v)
                if df is not None:
                    return df
            elif isinstance(v, dict):
                if "table" in v and isinstance(v["table"], pd.DataFrame):
                    df = _normalize_df(v["table"])
                    if df is not None:
                        return df
                try:
                    df = _normalize_df(pd.DataFrame(v))
                    if df is not None:
                        return df
                except Exception:
                    pass
            elif isinstance(v, (list, tuple)):
                try:
                    df = _normalize_df(pd.DataFrame(v))
                    if df is not None:
                        return df
                except Exception:
                    pass

    # 2) Fallback: scan every value for something that looks like a feature-importance DataFrame.
    for v in results.values():
        if isinstance(v, pd.DataFrame):
            df = _normalize_df(v)
            if df is not None:
                return df

    return None


def _render_fi_plot(fi_df: pd.DataFrame, topn: int = 10):
    """Plot TEXGISA Top-k as a horizontal bar chart (k<=10)."""
    if fi_df is None or fi_df.empty:
        st.info("No feature importance available.")
        return

    df = fi_df.copy()
    # Sort by importance so the chart reflects the expected "importance"/"directional_mean" schema.
    if "importance" in df.columns:
        df = df.sort_values("importance", ascending=False)
    # Limit the chart to the top-k rows (up to 10) for readability.
    k = min(topn, len(df))
    df_top = df.head(k)

    # Prepare labels and values for plotting.
    y_labels = list(reversed(df_top["feature"].astype(str).tolist()))
    x_vals   = list(reversed(df_top["importance"].astype(float).tolist()))

    # When directional information exists, colour bars by sign (blue = positive, red = negative).
    colors = None
    if "directional_mean" in df_top.columns:
        signs = df_top["directional_mean"].apply(lambda v: 1 if v >= 0 else -1).tolist()
        colors = ["#60a5fa" if s > 0 else "#f87171" for s in reversed(signs)]

    # Render the horizontal bar chart.
    import matplotlib.pyplot as plt
    fig_h = 4 + 0.35 * k  # Adjust the figure height based on the number of rows.
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(y_labels, x_vals, color=colors)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(f"TEXGISA Top-{k} Features (TEXGISA)")
    ax.grid(axis="x", alpha=0.2)
    st.pyplot(fig)
    # Textual explanation for FI Top-k
    _md_explain(_explain_plot("fi_topk", df=df_top))

def _inject_metrics_css():
    st.markdown(
        """
        <style>
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(220px,1fr)); gap: 12px; }
        .kpi-card {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.035);
        }
        .kpi-title { font-size: 0.85rem; opacity: 0.9; margin-bottom: 6px; }
        .kpi-value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                     font-size: 1.4rem; font-weight: 600; letter-spacing: 0.2px; }
        .kpi-sub { font-size: 0.8rem; opacity: 0.8; margin-top: 4px; }
        .kpi-bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,0.08); margin-top: 8px; overflow: hidden; }
        .kpi-bar > span { display:block; height: 100%; background: linear-gradient(90deg,#22d3ee,#a78bfa); }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_metrics_block(results: dict):
    # Gather every numeric metric from the results payload.
    items = []
    for k, v in results.items():
        if isinstance(v, (int, float, np.floating)):
            items.append((str(k), float(v)))
    if not items:
        return

    _inject_metrics_css()


    # Pretty-print common metrics while falling back to numeric formatting for others.
    def _fmt_val(name, val):
        if "index" in name.lower() or "c-index" in name.lower() or "cindex" in name.lower():
            return f"{val:.4f}"
        if name.lower().endswith(("epoch", "epochs", "samples", "bins", "events")) or abs(val - round(val)) < 1e-9:
            return f"{int(round(val)):,}"
        return f"{val:.4f}"

    # Render KPI cards with optional progress bars for c-index metrics.
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    for name, val in items:
        bar_html = ""
        if "index" in name.lower() or "c-index" in name.lower() or "cindex" in name.lower():
            p = max(0.0, min(1.0, val)) * 100.0
            bar_html = f'<div class="kpi-bar"><span style="width:{p:.2f}%"></span></div>'
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{name}</div>
                <div class="kpi-value">{_fmt_val(name, val)}</div>
                {bar_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Provide an expandable raw table for copy-and-paste workflows.
    with st.expander("See raw table", expanded=False):
        df = pd.DataFrame(items, columns=["Metric", "Value"]).sort_values("Metric", kind="stable")
        try:
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                height=min(320, 64 + 28 * len(df)),
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value":  st.column_config.NumberColumn("Value", format="%.4f", width="small"),
                },
            )
        except Exception:
            # Fallback for older Streamlit versions without column_config.
            st.table(df)
HAS_MYSA = True



def _ensure_binary_event(series, positive=1):
    """Map event column to 0/1 with given positive label."""
    s = series.copy()
    if s.dtype.kind in "biufc":
        # numeric-like: only map if not {0,1}
        uniq = sorted(pd.unique(s.dropna()))
        if set(uniq) == {0,1}:
            return s.astype(int)
        else:
            return (s == positive).astype(int)
    else:
        return (s == positive).astype(int)


def _build_expert_rules_from_editor(df_rules, important_features=None):
    """Convert data_editor dataframe to expert_rules dict expected by MySA."""

    rules: List[Dict[str, Any]] = []
    important_features = [str(f).strip() for f in (important_features or []) if str(f).strip()]
    result: Dict[str, Any] = {"rules": []}
    if important_features:
        result["important_features"] = important_features
    if df_rules is None or df_rules.empty:
        return result

    for _, row in df_rules.iterrows():
        feat = str(row.get("feature", "")).strip()
        if not feat:
            continue

        rule: Dict[str, Any] = {"feature": feat}

        # Back-compat: accept legacy column name `min_mag` while supporting `importance_floor`
        importance_floor = None
        if "importance_floor" in row:
            importance_floor = row.get("importance_floor")
        elif "min_mag" in row:
            importance_floor = row.get("min_mag")
        if pd.notna(importance_floor):
            floor_val = float(importance_floor)
            if floor_val > 0:
                rule["importance_floor"] = floor_val

        weight_val = row.get("weight", None)
        if pd.notna(weight_val):
            try:
                weight = float(weight_val)
            except (TypeError, ValueError):
                weight = None
            if weight is not None and weight > 0:
                rule["weight"] = weight

        # Additional legacy fields retained for compatibility with historical configurations
        relation = row.get("relation") if "relation" in row else None
        if isinstance(relation, str):
            relation = relation.strip() or None
        if relation and relation.lower() not in {"none", ""}:
            rule["relation"] = relation

        if "sign" in row:
            try:
                sign = int(row.get("sign", 0))
            except (TypeError, ValueError):
                sign = 0
            if sign != 0:
                rule["sign"] = sign

        if len(rule) > 1:
            rules.append(rule)

    result["rules"] = rules
    return result


def _canonicalize_id_column(df: Optional[pd.DataFrame], col: Optional[str]) -> Optional[pd.DataFrame]:
    if df is None or not col or col not in df.columns:
        return df
    df[col] = canonicalize_series(df[col])
    return df


def _build_multimodal_sources(config: dict) -> Optional[Dict[str, Any]]:
    dm = st.session_state.get("data_manager") if "data_manager" in st.session_state else None
    if dm is None:
        return None

    has_image = getattr(dm, "image_df", None) is not None
    has_sensor = getattr(dm, "sensor_df", None) is not None
    if not (has_image or has_sensor):
        return None

    id_col = st.session_state.get("mm_tab_id")
    if not id_col:
        return None

    sources: Dict[str, Any] = {"id_col": id_col}

    tab_df = getattr(dm, "tabular_df", None)
    if tab_df is not None:
        cfg_feats = config.get("feature_cols")
        if isinstance(cfg_feats, str):
            cfg_feats = [cfg_feats]
        elif cfg_feats is not None:
            try:
                cfg_feats = list(cfg_feats)
            except TypeError:
                cfg_feats = None

        exclude_cols = {id_col}
        time_col = config.get("time_col")
        event_col = config.get("event_col")
        if isinstance(time_col, str):
            exclude_cols.add(time_col)
        if isinstance(event_col, str):
            exclude_cols.add(event_col)

        if cfg_feats is None:
            tab_feats = [c for c in tab_df.columns if c not in exclude_cols]
        else:
            tab_cols = set(tab_df.columns)
            tab_feats = [c for c in cfg_feats if c in tab_cols and c not in exclude_cols]
            if not tab_feats:
                tab_feats = [c for c in tab_df.columns if c not in exclude_cols]

        sources["tabular"] = {
            "data": tab_df.copy(),
            "id_col": id_col,
            "feature_cols": tab_feats,
        }

    if has_image:
        img_df = dm.image_df
        img_id = st.session_state.get("mm_img_id") or id_col
        img_feats = [c for c in img_df.columns if c not in {img_id, "duration", "event"}]
        sources["image"] = {
            "data": img_df.copy(),
            "id_col": img_id,
            "feature_cols": img_feats,
        }
        raw_manifest = st.session_state.get("mm_image_manifest")
        raw_root = st.session_state.get("mm_image_root_raw")
        path_col = st.session_state.get("mm_image_path_col", "image")
        id_override = st.session_state.get("mm_image_id_col", img_id)
        if raw_manifest is not None and raw_root:
            sources["image"]["raw_assets"] = {
                "manifest": raw_manifest.copy(),
                "root": raw_root,
                "path_col": path_col,
                "id_col": id_override,
            }

    if has_sensor:
        sens_df = dm.sensor_df
        sens_id = st.session_state.get("mm_sens_id") or id_col
        sens_feats = [c for c in sens_df.columns if c not in {sens_id, "duration", "event"}]
        sources["sensor"] = {
            "data": sens_df.copy(),
            "id_col": sens_id,
            "feature_cols": sens_feats,
        }
        raw_manifest = st.session_state.get("mm_sensor_manifest")
        raw_root = st.session_state.get("mm_sensor_root_raw")
        path_col = st.session_state.get("mm_sensor_path_col", "file")
        id_override = st.session_state.get("mm_sensor_id_col", sens_id)
        if raw_manifest is not None and raw_root:
            sources["sensor"]["raw_assets"] = {
                "manifest": raw_manifest.copy(),
                "root": raw_root,
                "path_col": path_col,
                "id_col": id_override,
            }

    def _has_features(info: Any) -> bool:
        """Return True when a modality descriptor contains non-empty features."""
        if not isinstance(info, dict):
            return False
        if info.get("raw_assets"):
            return True
        cols = info.get("feature_cols")
        if cols is None:
            return False
        if isinstance(cols, (list, tuple)):
            return len(cols) > 0
        # Support pandas Index / Series etc.
        try:
            return bool(len(cols))
        except TypeError:
            return False

    extra_modalities = [k for k in ("image", "sensor") if _has_features(sources.get(k))]
    if not extra_modalities:
        return None

    return sources


def _plot_survival_curves(surv_df: pd.DataFrame, max_lines: int = 5):
    st.subheader("Predicted Survival Trajectories")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Draw up to max_lines curves for clarity
    cols = surv_df.columns[:max_lines]
    for pid in cols:
        ax.step(surv_df.index, surv_df[pid], where="post", label=str(pid))
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0.0, 1.05)
    if len(cols) > 1:
        ax.legend(title="Sample", loc="best")
    st.pyplot(fig)

    # Textual explanation
    try:
        last_vals = [float(surv_df[pid].iloc[-1]) for pid in cols]
    except Exception:
        last_vals = []
    _md_explain(_explain_plot("surv_traj", cols=list(cols), last_vals=last_vals))


def _render_training_curve_history(curve_data: Dict[str, Any]):
    entries: List[Dict[str, Any]] = curve_data.get("entries") or []
    if not entries:
        return

    time_bins = curve_data.get("time_bins")
    if not time_bins and entries and entries[0].get("hazards"):
        time_bins = list(range(1, len(entries[0]["hazards"][0]) + 1))
    time_bins = time_bins or list(range(1, 2))

    sample_ids: List[str] = curve_data.get("sample_ids") or []
    if not sample_ids and entries and entries[0].get("hazards"):
        sample_ids = [f"Sample {i + 1}" for i in range(len(entries[0]["hazards"]))]
    sample_meta = curve_data.get("sample_metadata") or [{} for _ in sample_ids]

    def _format_epoch(option: int) -> str:
        entry = entries[option]
        val_c = entry.get("val_cindex")
        if val_c is None or (isinstance(val_c, float) and math.isnan(val_c)):
            return f"Epoch {entry.get('epoch', option + 1)}"
        return f"Epoch {entry.get('epoch', option + 1)} (C-index={val_c:.3f})"

    epoch_idx = st.select_slider(
        "Select training epoch",
        options=list(range(len(entries))),
        value=len(entries) - 1,
        format_func=_format_epoch,
        key="training_curve_epoch",
    )

    current_entry = entries[epoch_idx]
    baseline_entry = entries[0]
    prev_entry = entries[epoch_idx - 1] if epoch_idx > 0 else current_entry

    def _ensure_2d(arr_like: Any) -> np.ndarray:
        arr = np.asarray(arr_like, dtype=float)
        if arr.size == 0:
            return np.zeros((len(sample_ids), len(time_bins)), dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr[: len(sample_ids), : len(time_bins)]

    hazard_current = _ensure_2d(current_entry.get("hazards", []))
    survival_current = _ensure_2d(current_entry.get("survival", []))
    n_samples = min(len(sample_ids), hazard_current.shape[0])
    sample_ids = sample_ids[:n_samples]
    sample_meta = sample_meta[:n_samples]
    hazard_current = hazard_current[:n_samples]
    survival_current = survival_current[:n_samples]

    hazard_prev = _ensure_2d(prev_entry.get("hazards", hazard_current))[:n_samples]
    survival_prev = _ensure_2d(prev_entry.get("survival", survival_current))[:n_samples]
    hazard_base = _ensure_2d(baseline_entry.get("hazards", hazard_current))[:n_samples]
    survival_base = _ensure_2d(baseline_entry.get("survival", survival_current))[:n_samples]

    palette = list(pq.Set2) + list(pq.Plotly) + list(pq.D3)
    color_cycle = itertools.cycle(palette)
    colors = [next(color_cycle) for _ in range(max(n_samples, 1))]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Hazard vs Time", "Survival vs Time"),
        shared_xaxes=True,
    )

    hazard_global_max = 0.0
    for entry in entries:
        hz = np.asarray(entry.get("hazards", []), dtype=float)
        if hz.size:
            hazard_global_max = max(hazard_global_max, float(np.nanmax(hz)))

    for idx in range(n_samples):
        label = sample_ids[idx]
        color = colors[idx]
        fig.add_trace(
            go.Scatter(
                x=time_bins,
                y=hazard_current[idx],
                name=f"{label} Hazard",
                legendgroup=label,
                line=dict(color=color, width=3),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_bins,
                y=survival_current[idx],
                name=f"{label} Survival",
                legendgroup=label,
                line=dict(color=color, dash="dash", width=3),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    frames = []
    for entry in entries:
        hz = _ensure_2d(entry.get("hazards", []))[:n_samples]
        sv = _ensure_2d(entry.get("survival", []))[:n_samples]
        frame_traces = []
        for idx in range(n_samples):
            frame_traces.append(
                go.Scatter(x=time_bins, y=hz[idx])
            )
        for idx in range(n_samples):
            frame_traces.append(
                go.Scatter(x=time_bins, y=sv[idx])
            )
        frames.append(
            go.Frame(
                data=frame_traces,
                name=str(entry.get("epoch", "")),
            )
        )

    fig.frames = frames

    slider_steps = [
        {
            "args": [[str(entry.get("epoch", ""))], {"frame": {"duration": 600, "redraw": False}, "mode": "immediate"}],
            "label": f"Ep {entry.get('epoch', idx + 1)}",
            "method": "animate",
        }
        for idx, entry in enumerate(entries)
    ]

    fig.update_layout(
        height=520,
        margin=dict(t=60, l=60, r=40, b=50),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.18, x=0.0),
        xaxis=dict(title="Time bin"),
        xaxis2=dict(title="Time bin"),
        yaxis=dict(title="Instantaneous hazard", range=[0, max(0.05, hazard_global_max * 1.1)]),
        yaxis2=dict(title="Survival probability", range=[0, 1.02]),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.18,
                "showactive": False,
                "pad": {"r": 10, "t": 30},
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 600, "redraw": False}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": epoch_idx,
                "pad": {"t": 30},
                "steps": slider_steps,
                "currentvalue": {"prefix": "Epoch: "},
            }
        ],
    )

    if fig.layout.sliders:
        fig.layout.sliders[0]["active"] = epoch_idx

    col_plot, col_text = st.columns([0.65, 0.35], gap="large")
    with col_plot:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Use the slider or play controls to follow how hazards and survival curves evolve across epochs.")

    val_c = current_entry.get("val_cindex")
    prev_c = prev_entry.get("val_cindex")
    base_c = baseline_entry.get("val_cindex")

    def _safe_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            if math.isnan(value):
                return None
        except TypeError:
            return None
        return float(value)

    val_c = _safe_float(val_c)
    prev_c = _safe_float(prev_c)
    base_c = _safe_float(base_c)

    mean_hazard_cur = float(np.nanmean(hazard_current)) if hazard_current.size else float("nan")
    mean_hazard_prev = float(np.nanmean(hazard_prev)) if hazard_prev.size else mean_hazard_cur
    hazard_delta_prev = mean_hazard_cur - mean_hazard_prev

    tail_surv_cur = float(np.nanmean(survival_current[:, -1])) if survival_current.size else float("nan")
    tail_surv_prev = float(np.nanmean(survival_prev[:, -1])) if survival_prev.size else tail_surv_cur
    tail_surv_base = float(np.nanmean(survival_base[:, -1])) if survival_base.size else tail_surv_cur
    tail_delta_prev = tail_surv_cur - tail_surv_prev
    tail_delta_base = tail_surv_cur - tail_surv_base

    lines: List[str] = []
    if val_c is not None:
        if prev_c is not None and epoch_idx > 0:
            diff = val_c - prev_c
            if diff > 0:
                delta_txt = f"+{diff:.3f} vs previous epoch"
            elif diff < 0:
                delta_txt = f"{diff:.3f} vs previous epoch"
            else:
                delta_txt = "no change vs previous epoch"
            lines.append(f"Validation C-index: {val_c:.3f} ({delta_txt}).")
        elif base_c is not None and epoch_idx > 0:
            diff = val_c - base_c
            if diff > 0:
                delta_txt = f"+{diff:.3f} vs first epoch"
            elif diff < 0:
                delta_txt = f"{diff:.3f} vs first epoch"
            else:
                delta_txt = "no change vs first epoch"
            lines.append(f"Validation C-index: {val_c:.3f} ({delta_txt}).")
        else:
            lines.append(f"Validation C-index: {val_c:.3f}.")

    if not math.isnan(mean_hazard_cur) and not math.isnan(mean_hazard_prev):
        if abs(hazard_delta_prev) < 1e-9:
            change_clause = "matching the previous epoch."
        else:
            direction = "decreased" if hazard_delta_prev < 0 else "increased"
            change_clause = f"{direction} by {abs(hazard_delta_prev):.3f} compared with the previous epoch."
        lines.append(
            f"Mean instantaneous hazard across monitored samples is {mean_hazard_cur:.3f}, {change_clause}"
        )

    if not math.isnan(tail_surv_cur) and not math.isnan(tail_surv_base):
        if abs(tail_delta_base) < 1e-9:
            baseline_clause = "matching the first epoch"
            interpret = "suggesting overall risk remains steady."
        elif tail_delta_base > 0:
            baseline_clause = f"+{abs(tail_delta_base):.3f} vs the first epoch"
            interpret = "suggesting the model is suppressing risk over time."
        else:
            baseline_clause = f"-{abs(tail_delta_base):.3f} vs the first epoch"
            interpret = "suggesting risk is accumulating faster."
        lines.append(
            f"Mean survival at the horizon is {tail_surv_cur:.3f} ({baseline_clause}), {interpret}"
        )

    survival_end = survival_current[:, -1] if survival_current.size else np.array([])
    if survival_end.size:
        try:
            worst_idx = int(np.nanargmin(survival_end))
        except ValueError:
            worst_idx = None
        if worst_idx is not None and 0 <= worst_idx < n_samples:
            worst_label = sample_ids[worst_idx]
            worst_surv = float(survival_end[worst_idx])
            worst_hazard_peak = float(np.nanmax(hazard_current[worst_idx]))
            lines.append(
                f"{worst_label} finishes with survival ≈ {worst_surv:.3f} and a peak hazard ≈ {worst_hazard_peak:.3f}; "
                "track this individual to understand how training shifts high-risk profiles."
            )

    with col_text:
        st.markdown("**Training trajectory highlights**")
        if lines:
            st.markdown("\n".join(f"- {ln}" for ln in lines))
        else:
            st.markdown("No curve statistics are available yet to summarise this epoch.")

        durations = [meta.get("duration") for meta in sample_meta]
        events = [meta.get("event") for meta in sample_meta]
        avg_hazard = np.nanmean(hazard_current, axis=1) if hazard_current.size else np.array([])
        table_dict = {
            "Sample": sample_ids,
            "Duration": durations,
            "Event": events,
            "End survival": np.round(survival_end, 3) if survival_end.size else [],
            "Mean hazard": np.round(avg_hazard, 3) if avg_hazard.size else [],
        }
        table_df = pd.DataFrame(table_dict)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

def _compute_km(durations, events, limit_to_last_event=True, bin_width=0):
    """
    durations: 1D array-like of times (float/int)
    events   : 1D array-like of {0,1}
    limit_to_last_event: if True, truncate the timeline to the last observed event (later points are censored only).
    bin_width: when >0, bucket times using this width (for example monthly or yearly bins) and use the bin endpoint as the time coordinate.
    Returns (t, S) where t contains only event-time nodes so the staircase curve remains clear.
    """
    import numpy as np
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)

    if bin_width and bin_width > 0:
        # Map each time to the right edge of its bin for clearer presentation.
        d = (np.floor(d / bin_width) * bin_width).astype(float)

    if limit_to_last_event and (e == 1).any():
        t_last_event = d[e == 1].max()
        keep = d <= t_last_event
        d, e = d[keep], e[keep]

    # Sort by time.
    order = np.argsort(d)
    d, e = d[order], e[order]

    uniq = np.unique(d)
    n_at_risk = len(d)
    S = 1.0
    curve_t = [0.0]
    curve_S = [1.0]

    # Update S only at event times; pure censoring reduces the risk set without changing survival.
    for tt in uniq:
        at_t = (d == tt)
        m = at_t.sum()                # Total individuals (events + censoring) at time tt.
        d_events = (e[at_t] == 1).sum()  # Number of events at time tt.
        if d_events > 0 and n_at_risk > 0:
            S = S * (1.0 - d_events / n_at_risk)
            curve_t.append(float(tt))
            curve_S.append(float(S))
        n_at_risk -= m

    return np.array(curve_t), np.array(curve_S)


def _plot_km_overall(df, limit_to_last_event=True, bin_width=0):
    import matplotlib.pyplot as plt
    t, S = _compute_km(
        df["duration"].values,
        df["event"].values,
        limit_to_last_event=limit_to_last_event,
        bin_width=bin_width,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(t, S, where="post")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan–Meier (overall)")
    ax.set_ylim(0.0, 1.05)
    st.pyplot(fig)
    # Textual explanation
    _md_explain(_explain_plot("km_overall", t=t, S=S))


def _plot_km_by_risk(df, surv_df, n_groups=3, limit_to_last_event=True, bin_width=0):
    import numpy as np, matplotlib.pyplot as plt
    if not isinstance(surv_df, pd.DataFrame) or surv_df.empty:
        st.info("No predictions available for risk stratification. Run a model first.")
        return
    last_surv = surv_df.iloc[-1].values
    risk = 1.0 - np.asarray(last_surv)

    # Align indices with the prediction DataFrame when possible.
    try:
        cols = surv_df.columns
        if np.issubdtype(np.array(cols).dtype, np.number):
            idx = np.array(cols, dtype=int)
        else:
            idx = cols
        durations = df.loc[idx, "duration"].values
        events = df.loc[idx, "event"].values
    except Exception:
        durations = df["duration"].values[: len(risk)]
        events = df["event"].values[: len(risk)]

    q = np.quantile(risk, [1/3, 2/3])
    groups = np.digitize(risk, q, right=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    S_end_per_group = []
    for g in range(3):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        t, S = _compute_km(
            durations[mask], events[mask],
            limit_to_last_event=limit_to_last_event,
            bin_width=bin_width
        )
        ax.step(t, S, where="post", label=f"Risk group {g+1}")
        # record last survival for explanation
        S_end_per_group.append(float(S[-1]) if len(S) else float("nan"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan–Meier by risk groups")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    st.pyplot(fig)

    # Textual explanation
    _md_explain(_explain_plot("km_groups", S_end=S_end_per_group))



def show():
    st.title("Survival Analysis Workbench")
    st.caption("You are here: Model Training")
    # === New: No-path image wizard (ZIP → labels → features → DataFrame/Train) ===
    from image_pipeline import (
        images_to_dataframe,
        build_manifest_from_zip,
        manifest_template_csv_bytes,
        unzip_images_to_temp,
    )

    # === New: Image ZIP wizard (ResNet-50 -> Features) =============================
    with st.expander("🖼️ Build Training Data from Image Data (ResNet-50 → Features)", expanded=False):
        st.markdown(
            "Steps: Upload **Image ZIP** → Complete/Import **Survival Labels** → One-click generate data table and directly train TEXGISA.\n\n"
            "**Label Definitions**: `duration` is the follow-up time, `event` indicates if the event occurred (0=No, 1=Yes).\n\n"
            "**Multimodal note**: Only TEXGISA consumes the raw images for end-to-end multimodal optimisation; other algorithms will use the generated tabular feature table."
        )

        # 1) Upload ZIP (Required)
        zip_up = uploader_with_help(
            "① Upload Image ZIP (Required)",
            key="img_zip_simple",
            help_text=CHANNEL_HELP_TEXT["img_zip_simple"],
            type=["zip"],
        )

        # 2) Parse ZIP and generate manifest (no path concept)
        if "imgwiz_manifest" not in st.session_state:
            st.session_state["imgwiz_manifest"] = None
            st.session_state["imgwiz_root"] = ""

        if st.button("Parse ZIP and Generate Labeling Manifest", use_container_width=True):
            if zip_up is None:
                st.error("Please upload an Image ZIP file first.")
            else:
                try:
                    mdf, root = build_manifest_from_zip(zip_up)
                    st.session_state["imgwiz_manifest"] = mdf
                    st.session_state["imgwiz_root"] = root
                    st.success(f"Parsed {len(mdf)} images. Next step: Complete `duration`/`event`.")
                except Exception as e:
                    st.error(f"Parsing failed: {e}")

        mdf = st.session_state.get("imgwiz_manifest", None)
        if mdf is not None:
            st.divider()
            st.subheader("② Fill in or Import Labels")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.download_button(
                    "Download Label Template CSV",
                    data=manifest_template_csv_bytes(mdf),
                    file_name="image_labels_template.csv",
                    mime="text/csv",
                    help="The template includes all image filenames. Please fill in the values in the duration/event columns."
                )
            with c2:
                labels_up = uploader_with_help(
                    "Or upload your completed label CSV",
                    key="img_labels_csv",
                    help_text=CHANNEL_HELP_TEXT["img_labels_csv"],
                    type=["csv"],
                )

            # If a label CSV is uploaded, merge it automatically
            if labels_up is not None:
                try:
                    lab = pd.read_csv(labels_up)
                    lab.columns = [c.strip().lower() for c in lab.columns]
                    # For compatibility with 'filename' naming
                    if "filename" in lab.columns and "image" not in lab.columns:
                        lab = lab.rename(columns={"filename": "image"})
                    for col in ("image", "duration", "event"):
                        if col not in lab.columns:
                            raise KeyError(f"CSV is missing column: {col}")
                    base = mdf.drop(columns=[c for c in ["duration","event"] if c in mdf.columns], errors="ignore")
                    mdf = base.merge(lab[["image","duration","event"]], on="image", how="left")
                    st.session_state["imgwiz_manifest"] = mdf
                    st.success("Successfully merged the label CSV. You can also continue editing in the table below.")
                except Exception as e:
                    st.error(f"Merge failed: {e}")

            # Also allow direct editing in the interface
            edited = st.data_editor(
                st.session_state["imgwiz_manifest"],
                num_rows="dynamic",
                use_container_width=True,
                key="imgwiz_editor",
                hide_index=True
            )
            st.session_state["imgwiz_manifest"] = edited

            # Label validation
            def _validate(df: pd.DataFrame) -> str:
                need = {"image", "duration", "event"}
                if not need.issubset(df.columns):
                    return f"Manifest is missing columns: {need - set(df.columns)}"
                dur = pd.to_numeric(df["duration"], errors="coerce")
                evt = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype("Int64")
                if dur.isna().any():
                    return "There are still empty or non-numeric `duration` values."
                if (~evt.isin([0,1])).any():
                    return "`event` must be 0 or 1."
                return ""

            err = _validate(edited)
            if err:
                st.warning(f"Please complete the labels: {err}")
                st.stop()

            st.divider()
            st.subheader("③ Generate Training Table (and optionally run a quick test with TEXGISA)")
            c3, c4, c5 = st.columns(3)
            with c3:
                    backbone = st.selectbox(
                        "Feature Backbone Network",
                        ["resnet50", "resnet34", "resnet18"],
                        index=0,
                        help="ResNet feature extractor used for embeddings. (ViT/CLIP can be added after validation.)",
                    )
            with c4:
                img_bs = st.number_input(
                    "Feature Extraction Batch Size",
                    8,
                    256,
                    64,
                    step=8,
                    help="Number of images processed per batch while extracting embeddings.",
                )
            with c5:
                img_workers = st.number_input(
                    "DataLoader Parallel Workers",
                    0,
                    8,
                    2,
                    step=1,
                    help="Background worker count for the feature-extraction data loader.",
                )

            if st.button("👉 Generate Data Table (duration/event + 2048-dim image features)", use_container_width=True):
                try:
                    with st.spinner("Extracting image features and building DataFrame..."):
                        df_img = images_to_dataframe(
                            edited,
                            image_root=st.session_state["imgwiz_root"],
                            model_name=backbone,
                            batch_size=int(img_bs),
                            num_workers=int(img_workers),
                        )
                    st.session_state["clinical_data"] = df_img
                    if "data_manager" in st.session_state:
                        dm = st.session_state.data_manager
                        dm.load_data(df_img, f"images+{backbone}")
                        dm.load_multimodal_data(image_df=df_img)
                    st.success(f"✅ Generated: {df_img.shape[0]} rows × {df_img.shape[1]} columns")
                    _preview_dataframe(df_img)
                except Exception as e:
                    st.error(f"Failed: {e}")

            # Optional: One-click quick run with MySA (small epochs for non-technical user experience)
            try:
                from models.mysa import run_mysa as _quick_mysa
            except Exception:
                _quick_mysa = None

            if _quick_mysa is not None and st.session_state.get("clinical_data") is not None:
                if st.button("🚀 One-Click Test Run with TEXGISA (Quick Metrics Preview)", use_container_width=True):
                    cfg = {
                        "n_bins": 20,
                        "val_ratio": 0.2,
                        "batch_size": 128,
                        "epochs": 5,          # Small number of epochs for quick feedback
                        "hidden": 128,
                        "depth": 2,
                        "dropout": 0.2,
                        "lambda_smooth": 0.01,
                        "lambda_expert": 0.0,
                        "num_workers": 0,
                    }
                    with st.spinner("Training TEXGISA (Quick Preview)..."):
                        res = _quick_mysa(st.session_state["clinical_data"], cfg)
                    vci = res.get("val_cindex", None)
                    tci = res.get("train_cindex", None)
                    st.success(f"Training complete. Train C-index={tci:.3f}, Val C-index={vci:.3f}" if vci is not None else "Training complete.")
                    if "survival_df" in res:
                        st.line_chart(res["survival_df"])
    # === End of new wizard block ===================================================

    # === New: Sensor ZIP wizard (full-sequence features, no sliding window) =======
    from sensor_pipeline import (
        build_manifest_from_sensors_zip,
        manifest_template_csv_bytes as manifest_template_csv_bytes_sensor,
        sensors_to_dataframe,
        unzip_sensors_to_temp,
    )

    with st.expander("📈 Build Training Data from Sensor Data (Full-sequence Features)", expanded=False):
        st.markdown(
            "Steps: Upload **Sensor ZIP** (one file per sample, CSV/Parquet) → Complete/Import `file,duration,event` → "
            "Extract **full-sequence features** at once → Generate training table and directly train TEXGISA.\n\n"
            "**Note**: By default, statistical and frequency-domain features are extracted from the entire sequence without a sliding window. If files contain timestamps, you can select a resampling frequency.\n\n"
            "**Multimodal note**: End-to-end optimisation on raw sensor waveforms is exclusive to TEXGISA; other algorithms will consume only the generated tabular features."
        )

        zip_up_sens = uploader_with_help(
            "① Upload Sensor ZIP (Required)",
            key="sensor_zip",
            help_text=CHANNEL_HELP_TEXT["sensor_zip"],
            type=["zip"],
        )
        if "senswiz_manifest" not in st.session_state:
            st.session_state["senswiz_manifest"] = None
            st.session_state["senswiz_root"] = ""

        if st.button("Parse ZIP and Generate Labeling Manifest (file,duration,event)", use_container_width=True):
            if zip_up_sens is None:
                st.error("Please upload a Sensor ZIP file first.")
            else:
                try:
                    mdf_s, root_s = build_manifest_from_sensors_zip(zip_up_sens)
                    st.session_state["senswiz_manifest"] = mdf_s
                    st.session_state["senswiz_root"] = root_s
                    st.success(f"Parsed {len(mdf_s)} sample files. Next step: Complete `duration`/`event`.")
                except Exception as e:
                    st.error(f"Parsing failed: {e}")

        mdf_s = st.session_state.get("senswiz_manifest", None)
        if mdf_s is not None:
            st.divider()
            st.subheader("② Fill in or Import Labels (file, duration, event)")

            cc1, cc2 = st.columns([1,1])
            with cc1:
                st.download_button(
                    "Download Label Template CSV (file,duration,event)",
                    data=manifest_template_csv_bytes_sensor(mdf_s),
                    file_name="sensor_labels_template.csv",
                    mime="text/csv",
                )
            with cc2:
                labels_up = uploader_with_help(
                    "Upload Your Completed Label CSV",
                    key="sensor_labels_csv",
                    help_text=CHANNEL_HELP_TEXT["sensor_labels_csv"],
                    type=["csv"],
                )

            # Merge label CSV
            if labels_up is not None:
                try:
                    lab = pd.read_csv(labels_up)
                    lab.columns = [c.strip().lower() for c in lab.columns]
                    for col in ("file","duration","event"):
                        if col not in lab.columns:
                            raise KeyError(f"CSV is missing column: {col}")
                    base = mdf_s.drop(columns=[c for c in ["duration","event"] if c in mdf_s.columns], errors="ignore")
                    mdf_s = base.merge(lab[["file","duration","event"]], on="file", how="left")
                    st.session_state["senswiz_manifest"] = mdf_s
                    st.success("Successfully merged the label CSV. You can also continue editing in the table below.")
                except Exception as e:
                    st.error(f"Merge failed: {e}")

            edited = st.data_editor(
                st.session_state["senswiz_manifest"],
                num_rows="dynamic",
                use_container_width=True,
                key="senswiz_editor",
                hide_index=True
            )
            st.session_state["senswiz_manifest"] = edited

            # Validate or allow using only labeled rows
            use_only_labeled = st.checkbox(
                "Use only samples with completed labels",
                value=True,
                help="Drop rows that still contain missing duration or event labels before processing.",
            )
            if use_only_labeled:
                edited = edited[pd.to_numeric(edited["duration"], errors="coerce").notna()]
                edited = edited[pd.to_numeric(edited["event"], errors="coerce").isin([0,1])]

            def _validate(df: pd.DataFrame) -> str:
                need = {"file","duration","event"}
                if not need.issubset(df.columns):
                    return f"Manifest is missing columns: {need - set(df.columns)}"
                dur = pd.to_numeric(df["duration"], errors="coerce")
                evt = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype("Int64")
                if dur.isna().any():
                    return "There are still empty or non-numeric `duration` values."
                if (~evt.isin([0,1])).any():
                    return "`event` must be 0 or 1."
                if len(df) == 0:
                    return "No available samples."
                return ""

            err = _validate(edited)
            if err:
                st.warning(f"Please complete the labels: {err}")
                st.stop()

            st.divider()
            st.subheader("③ Extract Full-Sequence Features and Generate Training Table")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                resample_hz = st.number_input("Resampling Frequency Hz (0=No Resampling)", min_value=0, max_value=512, value=0, step=1,
                                            help="Available when files contain timestamps; a uniform sampling rate is better for frequency analysis.")
            with sc2:
                max_rows = st.number_input("Max Rows to Read per File (0=Unlimited)", min_value=0, max_value=2_000_000, value=0, step=1000,
                                        help="Provides a safeguard for very large files.")
            with sc3:
                pass

            if st.button("👉 Extract Full-Sequence Features & Generate DataFrame", use_container_width=True):
                try:
                    with st.spinner("Calculating full-sequence statistical and frequency-domain features..."):
                        df_sens = sensors_to_dataframe(
                            edited,
                            sensor_root=st.session_state["senswiz_root"],
                            resample_hz=float(resample_hz),
                            max_rows_per_file=int(max_rows),
                        )
                    st.session_state["clinical_data"] = df_sens
                    if "data_manager" in st.session_state:
                        dm = st.session_state.data_manager
                        dm.load_data(df_sens, f"sensors_fullseq")
                        dm.load_multimodal_data(sensor_df=df_sens)
                    st.success(f"✅ Generated: {df_sens.shape[0]} rows × {df_sens.shape[1]} columns")
                    _preview_dataframe(df_sens)
                except Exception as e:
                    st.error(f"Failed: {e}")
    # === End of sensor wizard ======================================================


    # ===================== Multimodal data upload =============================
    with st.expander("🗂 Multimodal Data Upload", expanded=False):
        st.info(
            "TEXGISA is the only algorithm that performs end-to-end multimodal training. "
            "CoxTime, DeepSurv, and DeepHit will rely on the merged tabular table created in this section."
        )
        mode = st.radio(
            "Choose upload mode",
            ("Processed feature CSVs", "Raw assets (ZIP + manifest)"),
            key="mm_upload_mode",
            horizontal=True,
        )

        if mode == "Processed feature CSVs":
            tab_up = uploader_with_help(
                "Tabular CSV",
                key="mm_tabular",
                help_text=CHANNEL_HELP_TEXT["mm_tabular"],
                type=["csv"],
            )
            img_up = uploader_with_help(
                "Image CSV",
                key="mm_image",
                help_text=CHANNEL_HELP_TEXT["mm_image"],
                type=["csv"],
            )
            sens_up = uploader_with_help(
                "Sensor CSV",
                key="mm_sensor",
                help_text=CHANNEL_HELP_TEXT["mm_sensor"],
                type=["csv"],
            )

            if tab_up is not None:
                tab_df = pd.read_csv(tab_up)
                st.session_state["mm_tabular_df"] = tab_df
                _preview_dataframe(tab_df)
            if img_up is not None:
                img_df = pd.read_csv(img_up)
                st.session_state["mm_image_df"] = img_df
                _preview_dataframe(img_df)
            if sens_up is not None:
                sens_df = pd.read_csv(sens_up)
                st.session_state["mm_sensor_df"] = sens_df
                _preview_dataframe(sens_df)

            has_any = any(
                st.session_state.get(k) is not None for k in ["mm_tabular_df", "mm_image_df", "mm_sensor_df"]
            )
            if has_any:
                st.markdown("### Field Alignment")
                tab_id = img_id = sens_id = None
                if "mm_tabular_df" in st.session_state:
                    tab_id = st.selectbox(
                        "Tabular ID column",
                        st.session_state["mm_tabular_df"].columns,
                        key="mm_tab_id",
                        help="Identifier column in the processed tabular table used to align other modalities.",
                    )
                if "mm_image_df" in st.session_state:
                    img_id = st.selectbox(
                        "Image ID column",
                        st.session_state["mm_image_df"].columns,
                        key="mm_img_id",
                        help="Identifier column in the processed image table that matches the shared ID.",
                    )
                if "mm_sensor_df" in st.session_state:
                    sens_id = st.selectbox(
                        "Sensor ID column",
                        st.session_state["mm_sensor_df"].columns,
                        key="mm_sens_id",
                        help="Identifier column in the processed sensor table that matches the shared ID.",
                    )

                if st.button("Load Multimodal Data", key="mm_load_processed"):
                    base_tab = st.session_state.get("mm_tabular_df")
                    if base_tab is None:
                        st.warning("Tabular data is required for alignment.")
                    else:
                        tab_df = base_tab.copy()
                        _canonicalize_id_column(tab_df, tab_id)
                        st.session_state["mm_tabular_df"] = tab_df
                        combined = tab_df
                        if "mm_image_df" in st.session_state and img_id:
                            img_df = st.session_state["mm_image_df"].copy()
                            _canonicalize_id_column(img_df, img_id)
                            st.session_state["mm_image_df"] = img_df
                            combined = combined.merge(
                                img_df,
                                left_on=tab_id,
                                right_on=img_id,
                                how="left",
                            )
                        if "mm_sensor_df" in st.session_state and sens_id:
                            sens_df = st.session_state["mm_sensor_df"].copy()
                            _canonicalize_id_column(sens_df, sens_id)
                            st.session_state["mm_sensor_df"] = sens_df
                            combined = combined.merge(
                                sens_df,
                                left_on=tab_id,
                                right_on=sens_id,
                                how="left",
                            )

                        st.session_state["clinical_data"] = combined
                        if "data_manager" in st.session_state:
                            dm = st.session_state.data_manager
                            dm.load_multimodal_data(
                                tabular_df=st.session_state.get("mm_tabular_df"),
                                image_df=st.session_state.get("mm_image_df"),
                                sensor_df=st.session_state.get("mm_sensor_df"),
                            )
                            dm.load_data(combined, "multimodal_combined")
                        st.success(
                            f"✅ Loaded multimodal data ({combined.shape[0]} rows × {combined.shape[1]} columns)"
                        )
                        _preview_dataframe(combined)
        else:
            st.markdown(
                "Upload the raw assets (ZIP + manifest) exported by the simulator or your own pipeline."
            )
            tab_up = uploader_with_help(
                "Tabular CSV (required)",
                key="mm_raw_tabular",
                help_text=CHANNEL_HELP_TEXT["mm_raw_tabular"],
                type=["csv"],
            )
            id_col = st.text_input(
                "Common ID column for alignment",
                value=st.session_state.get("mm_raw_id_col", "id"),
                key="mm_raw_id_col",
                help="Identifier present in every modality and used to align subjects across tables and manifests.",
            )
            img_id_col = st.text_input(
                "Image manifest ID column",
                value=st.session_state.get("mm_raw_img_id_col", "id"),
                key="mm_raw_img_id_col",
                help="Column name inside the image manifest that matches the shared identifier.",
            )
            sens_id_col = st.text_input(
                "Sensor manifest ID column",
                value=st.session_state.get("mm_raw_sens_id_col", "id"),
                key="mm_raw_sens_id_col",
                help="Column name inside the sensor manifest that matches the shared identifier.",
            )

            c1, c2 = st.columns(2)
            with c1:
                img_zip = uploader_with_help(
                    "Image ZIP",
                    key="mm_raw_img_zip",
                    help_text=CHANNEL_HELP_TEXT["mm_raw_img_zip"],
                    type=["zip"],
                )
                img_manifest = uploader_with_help(
                    "Image manifest CSV",
                    key="mm_raw_img_manifest",
                    help_text=CHANNEL_HELP_TEXT["mm_raw_img_manifest"],
                    type=["csv"],
                )
                img_bs = st.number_input(
                    "Image batch size",
                    min_value=8,
                    max_value=256,
                    value=int(st.session_state.get("mm_raw_img_bs", 32)),
                    step=8,
                    key="mm_raw_img_bs",
                    help="Batch size used when extracting image embeddings from the raw ZIP archive.",
                )
            with c2:
                sens_zip = uploader_with_help(
                    "Sensor ZIP",
                    key="mm_raw_sensor_zip",
                    help_text=CHANNEL_HELP_TEXT["mm_raw_sensor_zip"],
                    type=["zip"],
                )
                sens_manifest = uploader_with_help(
                    "Sensor manifest CSV",
                    key="mm_raw_sensor_manifest",
                    help_text=CHANNEL_HELP_TEXT["mm_raw_sensor_manifest"],
                    type=["csv"],
                )
                sens_resample = st.number_input(
                    "Sensor resample Hz (0 = no resample)",
                    min_value=0,
                    max_value=256,
                    value=int(st.session_state.get("mm_raw_sensor_resample", 0)),
                    step=1,
                    key="mm_raw_sensor_resample",
                    help="Target frequency for resampling raw sensor sequences; leave at 0 to keep native sampling.",
                )
                sens_max_rows = st.number_input(
                    "Max rows per sensor file (0 = all)",
                    min_value=0,
                    max_value=2_000_000,
                    value=int(st.session_state.get("mm_raw_sensor_maxrows", 0)),
                    step=1000,
                    key="mm_raw_sensor_maxrows",
                    help="Upper limit on rows read from each sensor file to prevent oversized loads (0 keeps all rows).",
                )

            if st.button("Process Raw Multimodal Assets", key="mm_process_raw"):
                if tab_up is None:
                    st.warning("Please upload the tabular CSV before processing raw assets.")
                    st.stop()

                try:
                    tab_df = pd.read_csv(tab_up)
                    tab_df.columns = [str(c).strip() for c in tab_df.columns]
                except Exception as exc:
                    st.error(f"Failed to read tabular CSV: {exc}")
                    st.stop()

                if id_col not in tab_df.columns:
                    st.error(f"Tabular CSV must contain the ID column '{id_col}'.")
                    st.stop()

                _canonicalize_id_column(tab_df, id_col)

                st.session_state["mm_image_df"] = None
                st.session_state["mm_sensor_df"] = None

                image_df = None
                if img_zip is not None and img_manifest is not None:
                    try:
                        img_manifest_df = pd.read_csv(img_manifest)
                        img_manifest_df.columns = [str(c).strip() for c in img_manifest_df.columns]
                        if img_id_col not in img_manifest_df.columns:
                            raise KeyError(f"Image manifest is missing column '{img_id_col}'.")
                        with st.spinner("Extracting image embeddings..."):
                            img_root = unzip_images_to_temp(img_zip)
                            image_df = images_to_dataframe(
                                img_manifest_df,
                                image_root=img_root,
                                id_col=img_id_col,
                                batch_size=int(img_bs),
                                num_workers=0,
                            )
                        st.session_state["mm_image_manifest"] = img_manifest_df.copy()
                        st.session_state["mm_image_root_raw"] = img_root
                        st.session_state["mm_image_path_col"] = "image"
                        st.session_state["mm_image_id_col"] = img_id_col
                        if img_id_col != id_col and image_df is not None and img_id_col in image_df.columns:
                            image_df = image_df.rename(columns={img_id_col: id_col})
                        if image_df is not None and id_col in image_df.columns:
                            _canonicalize_id_column(image_df, id_col)
                        st.session_state["mm_image_df"] = image_df
                        _preview_dataframe(image_df)
                    except Exception as exc:
                        st.error(f"Image processing failed: {exc}")
                        image_df = None

                sensor_df = None
                if sens_zip is not None and sens_manifest is not None:
                    try:
                        sens_manifest_df = pd.read_csv(sens_manifest)
                        sens_manifest_df.columns = [str(c).strip() for c in sens_manifest_df.columns]
                        if sens_id_col not in sens_manifest_df.columns:
                            raise KeyError(f"Sensor manifest is missing column '{sens_id_col}'.")
                        with st.spinner("Extracting sensor features..."):
                            sens_root = unzip_sensors_to_temp(sens_zip)
                            sensor_df = sensors_to_dataframe(
                                sens_manifest_df,
                                sensor_root=sens_root,
                                id_col=sens_id_col,
                                resample_hz=float(sens_resample),
                                max_rows_per_file=int(sens_max_rows),
                            )
                        st.session_state["mm_sensor_manifest"] = sens_manifest_df.copy()
                        st.session_state["mm_sensor_root_raw"] = sens_root
                        st.session_state["mm_sensor_path_col"] = "file"
                        st.session_state["mm_sensor_id_col"] = sens_id_col
                        if sens_id_col != id_col and sensor_df is not None and sens_id_col in sensor_df.columns:
                            sensor_df = sensor_df.rename(columns={sens_id_col: id_col})
                        if sensor_df is not None and id_col in sensor_df.columns:
                            _canonicalize_id_column(sensor_df, id_col)
                        st.session_state["mm_sensor_df"] = sensor_df
                        _preview_dataframe(sensor_df)
                    except Exception as exc:
                        st.error(f"Sensor processing failed: {exc}")
                        sensor_df = None

                st.session_state["mm_tabular_df"] = tab_df

                if image_df is None:
                    image_df = st.session_state.get("mm_image_df")
                if sensor_df is None:
                    sensor_df = st.session_state.get("mm_sensor_df")

                def _prep_for_merge(df, *, preserve_cols: Optional[Sequence[str]] = None):
                    if df is None or id_col not in df.columns:
                        return None
                    tmp = df.copy()
                    _canonicalize_id_column(tmp, id_col)
                    preserve: Set[str] = {c for c in (preserve_cols or []) if c and c in tmp.columns}
                    drop_cols = [c for c in ("duration", "event") if c in tmp.columns]
                    if drop_cols:
                        tmp = tmp.drop(columns=drop_cols)
                    feat_cols = [c for c in tmp.columns if c != id_col]
                    if not feat_cols:
                        return None
                    adaptive_numeric: List[str] = []
                    for col in feat_cols:
                        if col in preserve:
                            continue
                        converted = pd.to_numeric(tmp[col], errors="coerce")
                        # Preserve columns that lose all information when coerced (e.g. asset paths).
                        if converted.notna().sum() == 0 and tmp[col].notna().sum() > 0:
                            preserve.add(col)
                            continue
                        tmp[col] = converted
                        adaptive_numeric.append(col)
                    if tmp[id_col].duplicated().any():
                        agg_map: Dict[str, str] = {}
                        for col in feat_cols:
                            if col in preserve:
                                agg_map[col] = "first"
                            else:
                                agg_map[col] = "mean"
                        tmp = tmp.groupby(id_col, as_index=False).agg(agg_map)
                    # Ensure preserved columns keep their original dtype/order alongside numeric ones.
                    cols_order = [id_col] + [c for c in feat_cols if c in tmp.columns and c != id_col]
                    tmp = tmp.loc[:, [c for c in cols_order if c in tmp.columns]]
                    return tmp

                combined = tab_df.copy()
                img_preserve = [st.session_state.get("mm_image_path_col")] if "mm_image_path_col" in st.session_state else None
                img_merge = _prep_for_merge(image_df, preserve_cols=img_preserve)
                if img_merge is not None:
                    combined = combined.merge(img_merge, on=id_col, how="left")
                sens_preserve = [st.session_state.get("mm_sensor_path_col")] if "mm_sensor_path_col" in st.session_state else None
                sens_merge = _prep_for_merge(sensor_df, preserve_cols=sens_preserve)
                if sens_merge is not None:
                    combined = combined.merge(sens_merge, on=id_col, how="left")

                combined_display = combined
                if id_col in combined_display.columns:
                    combined_display = combined_display.drop(columns=[id_col])

                st.session_state["clinical_data"] = combined_display
                st.session_state["mm_tab_id"] = id_col

                if "data_manager" in st.session_state:
                    dm = st.session_state.data_manager
                    dm.load_multimodal_data(
                        tabular_df=tab_df,
                        image_df=image_df,
                        sensor_df=sensor_df,
                    )
                    dm.load_data(combined_display, "multimodal_combined_raw")

                st.success(
                    f"✅ Processed raw multimodal assets ({combined_display.shape[0]} rows × {combined_display.shape[1]} columns)"
                )
                if id_col in combined.columns:
                    st.caption(
                        "ℹ️ Identifier column dropped from the working table to avoid feeding string IDs into the model. "
                        "The original ID is still preserved internally for modality alignment."
                    )
                _preview_dataframe(combined_display)

    # ===================== 1) Data upload & preview ===========================
    with st.expander("📘 Step-by-Step Guide for Tabular Data", expanded=False):
        st.markdown(
            "- Upload a CSV file with **duration** and **event** columns, plus features.\n"
            "- Select columns and model, set hyperparameters, and run.\n"
            "- For **TEXGISA**, you can set **λ_smooth** and **λ_expert**, and edit **Expert Rules**."
        )

    uploaded = uploader_with_help(
        "Upload CSV",
        key="clinical_upload",
        help_text=CHANNEL_HELP_TEXT["tabular_csv"],
        type=["csv"],
    )
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.session_state["clinical_data"] = data
            if "data_manager" in st.session_state:
                dm = st.session_state.data_manager
                dm.load_data(data, uploaded.name)
                dm.load_multimodal_data(tabular_df=data)
            st.success(f"✅ Loaded '{uploaded.name}' ({data.shape[0]} rows, {data.shape[1]} cols)")
            st.dataframe(data.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")

    if "clinical_data" not in st.session_state:
        st.info("⬆️ Upload a dataset to start.")
        return

    data = st.session_state["clinical_data"]

    # ===================== 2) Column selection ================================
    with st.expander("🧭 Column Mapping", expanded=True):
        cols = list(data.columns)
        c1, c2 = st.columns(2)
        with c1:
            time_col = field_with_help(
                st.selectbox, "Duration column", "time_col",
                options=cols, index=0 if cols else None
            )
        with c2:
            event_col = field_with_help(
                st.selectbox, "Event column", "event_col",
                options=cols, index=1 if len(cols) > 1 else 0
            )
        feat_default = [c for c in cols if c not in {time_col, event_col}]
        features = field_with_help(
            st.multiselect, "Feature columns", "features",
            options=cols, default=feat_default
        )
        with st.columns(2)[0]:
            event_positive = field_with_help(
                st.text_input, "Event positive label (for mapping to 1)", "positive_label",
                value="1"
            )


    mm_tab_id = None
    if any(st.session_state.get(k) is not None for k in ["mm_image_df", "mm_sensor_df"]):
        mm_tab_id = st.session_state.get("mm_tab_id")
        if mm_tab_id and mm_tab_id in features:
            features = [f for f in features if f != mm_tab_id]

    # Ensure the feature list is unique while preserving order.
    if features:
        seen = set()
        features = [f for f in features if not (f in seen or seen.add(f))]

    available_cols = set(data.columns)

    missing_features = [f for f in features if f not in available_cols]
    if missing_features:
        st.warning(
            "The following selected feature columns are no longer present in the current dataset and will be ignored: "
            + ", ".join(missing_features)
        )
        features = [f for f in features if f in available_cols]
        st.session_state["features"] = features

    if time_col not in available_cols or event_col not in available_cols:
        st.error(
            "Selected duration/event columns are not available in the working table. Please re-select columns from the "
            "dropdowns above."
        )
        return

    mm_tab_id_present = bool(mm_tab_id and mm_tab_id in available_cols)


    # Build the working dataframe with required column names.
    try:
        # Select the requested columns and normalise their names.
        base_cols = features + [time_col, event_col]
        if mm_tab_id_present and mm_tab_id not in base_cols:
            base_cols.append(mm_tab_id)

        missing_base = [c for c in base_cols if c not in available_cols]
        if missing_base:
            st.error(
                "The following required columns are missing from the dataset: " + ", ".join(missing_base)
            )
            return

        base = data[base_cols].copy()
        df = base.rename(columns={time_col: "duration", event_col: "event"})

        id_series = None
        if mm_tab_id_present and mm_tab_id in df.columns:
            id_series = df[mm_tab_id].copy()

        # Map the event column to {0,1} while respecting the chosen positive label.
        df["event"] = _ensure_binary_event(
            df["event"],
            positive=type(df["event"].iloc[0])(event_positive)
        )

        # ========== 🧹 Auto-clean UI controls ==========
        st.markdown("### 🧹 Auto-clean Data")
        c0, c1, c2 = st.columns([0.45, 0.35, 0.20])
        with c0:
            auto_clean = st.checkbox(
                "Enable auto-clean (bool→0/1, drop high-NaN/constant, Z-score)",
                value=True,
                key="csv_autoclean",
                help="Automatically coerce booleans, drop high-missing columns, and normalise features before training.",
            )
        with c1:
            nan_thresh = st.slider(
                "Missing-value threshold to drop a column",
                min_value=0.0, max_value=0.9, value=0.30, step=0.05,
                key="csv_nan_thresh"
            )
        with c2:
            do_zscore = st.checkbox(
                "Z-score",
                value=True,
                key="csv_zscore",
                help="Standardise each retained feature to zero mean and unit variance after cleaning.",
            )

        # Helper functions scoped locally to avoid polluting the module namespace.
        import pandas as _pd
        BOOL_STR = {"true","false","t","f","yes","no","y","n","0","1"}
        def _is_bool_like(s: _pd.Series) -> bool:
            if s.dtype == bool:
                return True
            if _pd.api.types.is_integer_dtype(s) and set(_pd.Series(s).dropna().unique()) <= {0, 1}:
                return True
            if _pd.api.types.is_object_dtype(s):
                vals = {str(v).strip().lower() for v in _pd.Series(s).dropna().unique()}
                return bool(vals) and vals <= BOOL_STR
            return False
        def _to_bool01(s: _pd.Series) -> _pd.Series:
            if s.dtype == bool:
                return s.astype("float32")
            if _pd.api.types.is_integer_dtype(s) and set(_pd.Series(s).dropna().unique()) <= {0, 1}:
                return s.astype("float32")
            m = {"true":1,"t":1,"yes":1,"y":1,"1":1, "false":0,"f":0,"no":0,"n":0,"0":0}
            return s.astype(str).str.strip().str.lower().map(m).astype("float32")

        # ========== Apply the cleaning steps to the selected feature columns ==========
        if auto_clean:
            # Copy only the feature subset.
            X = df[features].copy()

            if X.columns.duplicated().any():
                dupes = list(dict.fromkeys(X.columns[X.columns.duplicated()]))
                st.warning(
                    "Duplicate feature names detected; keeping the first occurrence for each of: "
                    + ", ".join(map(str, dupes))
                )
                X = X.loc[:, ~X.columns.duplicated()].copy()
                features = list(X.columns)

            # 1) Convert boolean-like values to 0/1 and coerce other non-numeric values to floats.
            for f in list(X.columns):
                s = X[f]
                if _is_bool_like(s):
                    X[f] = _to_bool01(s)
                elif not np.issubdtype(s.dtype, np.number):
                    X[f] = _pd.to_numeric(s, errors="coerce")

            # 2) Keep numeric features only.
            X = X.select_dtypes(include=["number"])

            # 3) Drop columns with excessive missingness or near-zero variance.
            nan_ratio = X.isna().mean()
            std = X.std(ddof=0)
            keep_mask = (nan_ratio <= float(nan_thresh)) & (std > 1e-12)
            dropped_nan = int((nan_ratio > float(nan_thresh)).sum())
            dropped_const = int((std <= 1e-12).sum())
            X = X.loc[:, keep_mask].fillna(0.0)

            # 4) Optionally Z-score the remaining features.
            if do_zscore and X.shape[1] > 0:
                mu = X.mean(0)
                sigma = X.std(0).replace(0.0, 1.0)
                X = (X - mu) / sigma

            # 5) Reassemble the final training table (duration, event, and cleaned numeric features).
            parts = []
            if id_series is not None:
                parts.append(id_series.to_frame(name=mm_tab_id))
            parts.append(df[["duration", "event"]])
            parts.append(X.astype("float32"))
            df = _pd.concat(parts, axis=1)

            # Update the feature list so later configuration panels stay in sync.
            features = list(X.columns)

            st.info(
                f"Auto-cleaned → final table: {df.shape[0]} rows × {df.shape[1]} cols; "
                f"dropped high-NaN cols: {dropped_nan}, dropped constant/zero-var cols: {dropped_const}."
            )
            try:
                st.dataframe(df.head(), use_container_width=True)
            except Exception:
                pass
        else:
            # When auto-clean is disabled, still coerce non-numeric columns to numeric as a safety net.
            warned_dupe = False
            for f in features:
                series = df[f]
                if isinstance(series, pd.DataFrame):
                    if not warned_dupe:
                        st.warning(
                            "Duplicate feature names detected; only the first occurrence will be used during cleaning."
                        )
                        warned_dupe = True
                    series = series.iloc[:, 0]
                if not np.issubdtype(series.dtype, np.number):
                    df[f] = pd.to_numeric(series, errors="coerce").fillna(0.0)
                else:
                    df[f] = series

    except Exception as e:
        st.error(f"Column mapping/cleaning failed: {e}")
        return


    # ===================== 3) Algorithm & hyperparameters =====================
    st.subheader("⚙️ Algorithm & Training Configuration")
    st.caption(_qhelp_md("algo_panel"))
    algo = field_with_help(
        st.selectbox, "Algorithm", "algo",
        ["CoxTime", "DeepSurv", "DeepHit", "TEXGISA"]
    )

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        batch_size = field_with_help(
            st.number_input, "Batch size", "batch_size",
            16, 1024, 128, step=16
        )
    with hc2:
        epochs = field_with_help(
            st.number_input, "Epochs", "epochs",
            10, 2000, 200, step=10
        )
    with hc3:
        lr = field_with_help(
            st.number_input, "Learning rate", "lr",
            1e-5, 1e-1, 1e-3, step=1e-5, format="%.5f"
        )

    # DeepHit-specific
    config = {}
    if algo == "DeepHit":
        st.markdown("**DeepHit settings**")
        config["num_intervals"] = field_with_help(
            st.number_input, "Time intervals", "num_intervals",
            5, 200, 50
        )


    # ===================== 4) TEXGISA regularizers & expert rules ================
    if algo == "TEXGISA":
        st.markdown("### Regularizers")
        st.caption(_qhelp_md("texgisa_regularizers"))
        r1, r2 = st.columns(2)
        with r1:
            lambda_smooth = field_with_help(
                st.number_input, "λ_smooth (temporal smoothness)", "lambda_smooth",
                0.0, 1.0, 0.01, step=0.01, format="%.2f"
            )
        with r2:
            lambda_expert = field_with_help(
                st.number_input, "λ_expert (expert prior penalty)", "lambda_expert",
                0.0, 10.0, 0.10, step=0.05, format="%.2f"
            )

        st.markdown("### Expert Guidance")
        st.caption(_qhelp_md("texgisa_guidance"))
        # Important set I selector
        prev_imp = st.session_state.get("important_features", [])
        default_imp = [f for f in prev_imp if f in features]
        important_features = st.multiselect(
            "Important features (set I)",
            options=features,
            default=default_imp,
            key="important_features_selector",
            help=_qhelp_md("important_features"),
        )
        st.session_state["important_features"] = important_features

        st.caption(
            "Features in set I are protected by the expert penalty; other features are softly suppressed unless justified by TEXGISA."
        )

        st.markdown("#### Directional / magnitude constraints (optional)")
        st.caption(_qhelp_md("texgisa_constraints"))
        st.caption(
            "The current release only supports encouraging minimum magnitude per feature. "
        )

        # Seed table with a few editable rows prioritising expert-marked features
        base_rows = list(dict.fromkeys(important_features))
        if not base_rows:
            base_rows = [features[0]] if features else []
        while len(base_rows) < 5:
            base_rows.append("")

        seed = pd.DataFrame({
            "feature": base_rows,
            "importance_floor": [0.0] * len(base_rows),
            "weight": [1.0] * len(base_rows),
        })
        edited = st.data_editor(
            seed,
            column_config={
                "feature": st.column_config.SelectboxColumn("feature", options=features, help="Choose a feature"),
                "importance_floor": st.column_config.NumberColumn("minimum TEXGISA magnitude", step=0.01, format="%.3f"),
                "weight": st.column_config.NumberColumn("penalty weight", step=0.1, format="%.2f"),
            },
            num_rows="dynamic",
            use_container_width=True,
        )
        expert_rules = _build_expert_rules_from_editor(edited, important_features)

        # Advanced TEXGISA/Generator controls
        advanced_box = st.expander("Advanced TEXGISA / Generator settings", expanded=False)
        with advanced_box:
            st.caption(_qhelp_md("texgisa_advanced"))
            ig_steps = field_with_help(
                st.number_input, "IG steps (M)", "ig_steps",
                5, 200, 20
            )
            latent_dim = field_with_help(
                st.number_input, "Generator latent dim", "latent_dim",
                2, 128, 16
            )
            extreme_dim = field_with_help(
                st.number_input, "Extreme code dim", "extreme_dim",
                1, 8, 1
            )
            gen_epochs = field_with_help(
                st.number_input, "Generator epochs", "gen_epochs",
                10, 2000, 200, step=10
            )
            gen_batch = field_with_help(
                st.number_input, "Generator batch size", "gen_batch",
                32, 2048, 256, step=32
            )
            gen_lr = field_with_help(
                st.number_input, "Generator LR", "gen_lr",
                1e-5, 1e-1, 1e-3, step=1e-5, format="%.5f"
            )
            gen_alpha_dist = field_with_help(
                st.number_input, "Generator distance weight α", "gen_alpha_dist",
                0.01, 10.0, 1.0, step=0.05, format="%.2f"
            )
            ig_batch_samples = field_with_help(
                st.number_input, "IG samples per batch (B')", "ig_batch_samples",
                4, 512, 32
            )
            ig_time_subsample = field_with_help(
                st.number_input, "Time-bins per batch for TEXGISA (T')", "ig_time_subsample",
                1, 200, 8
            )



        # Attach to config
        config.update({
            "lambda_smooth": float(lambda_smooth),
            "lambda_expert": float(lambda_expert),
            "expert_rules": expert_rules,
            "ig_steps": int(ig_steps),
            "latent_dim": int(latent_dim),
            "extreme_dim": int(extreme_dim),
            "gen_epochs": int(gen_epochs),
            "gen_batch": int(gen_batch),
            "gen_lr": float(gen_lr),
            "gen_alpha_dist": float(gen_alpha_dist),
            "ig_batch_samples": int(ig_batch_samples),
            "ig_time_subsample": int(ig_time_subsample),
        })

    # Common config
    config.update({
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "feature_cols": features,
    })

    mm_sources = _build_multimodal_sources(config)
    if mm_sources:
        if algo == "TEXGISA":
            config["multimodal_sources"] = mm_sources
            st.success(
                "✅ Raw multimodal inputs detected. TEXGISA will optimise all modalities end-to-end with TEXGISA."
            )
        else:
            st.warning(
                f"Raw multimodal inputs are loaded, but {algo} will ignore the raw assets and train on the merged table only. "
                "Switch to TEXGISA for end-to-end multimodal optimisation."
            )

    # ===================== 5) Run =============================================
    preview_clicked = False
    train_clicked = False
    train_with_playback_clicked = False
    fast_expert = False
    run_clicked = False

    if algo == "TEXGISA":
        c_run1, c_run2, c_run3 = st.columns(3)
        with c_run1:
            preview_clicked = st.button(
                "👀 Preview FI (no expert priors)",
                use_container_width=True,
                help="Run a short TEXGISA pass (λ_expert=0) to preview TEXGISA feature importance (powered by TEXGISA) before full training.",
            )
        with c_run2:
            train_clicked = st.button(
                "🚀 Train with Expert Priors",
                use_container_width=True,
                help="Train TEXGISA with the configured expert rules and λ_expert penalty (fast path without hazard playback).",
            )
            st.caption("Fastest training path – skips per-epoch curve capture.")
            fast_expert = st.checkbox(
                "Fast expert mode (lighter generator & TEXGISA importance)",
                value=True,
                help=_qhelp_md("fast_mode"),
            )
        with c_run3:
            train_with_playback_clicked = st.button(
                "🎬 Train + Hazard Playback",
                use_container_width=True,
                help="Train TEXGISA and capture per-epoch hazard/survival curves for the playback widget (runs longer).",
            )
            st.caption("Captures per-epoch curves for visual playback. Expect longer runtimes.")
    else:
        run_clicked = st.button(
            f"🚀 Train {algo}",
            use_container_width=True,
            help=f"Train {algo} using the selected configuration.",
        )


    if preview_clicked:
        with st.spinner("Training briefly and computing attributions (λ_expert=0)..."):
            try:
                cfg = dict(config)
                cfg["lambda_expert"] = 0.0              # Disable expert penalty during the preview pass.
                # Use a smaller epoch count for fast previews.
                cfg["epochs"] = min(int(cfg.get("epochs", 200)), 50)
                cfg["capture_training_curves"] = False
                results = run_analysis(algo, df, cfg)
                st.session_state["results"] = results
                st.success("✅ FI preview done.")
            except Exception as e:
                st.error(f"Preview failed: {e}")

    if train_clicked:
        with st.spinner("Training with expert priors..."):
            try:
                # Create a shallow copy of the configuration so we can adjust preview-specific knobs.
                cfg = dict(config)

                # Reduce expensive settings when fast-expert mode is enabled.
                if fast_expert:
                    cfg["gen_epochs"] = min(cfg.get("gen_epochs", 200), 40)
                    cfg["ig_steps"] = min(cfg.get("ig_steps", 20), 10)
                    cfg["ig_batch_samples"] = min(cfg.get("ig_batch_samples", 32), 24)
                    cfg["ig_time_subsample"] = min(cfg.get("ig_time_subsample", 8), 6)

                cfg["capture_training_curves"] = False

                # Execute the training run with the potentially adjusted configuration.
                results = run_analysis(algo, df, cfg)

                st.session_state["results"] = results
                st.success("✅ Training completed.")

            except Exception as e:
                st.error(f"Run failed: {e}")

    if train_with_playback_clicked:
        with st.spinner("Training with hazard playback (captures per-epoch curves)..."):
            try:
                cfg = dict(config)

                if fast_expert:
                    cfg["gen_epochs"] = min(cfg.get("gen_epochs", 200), 40)
                    cfg["ig_steps"] = min(cfg.get("ig_steps", 20), 10)
                    cfg["ig_batch_samples"] = min(cfg.get("ig_batch_samples", 32), 24)
                    cfg["ig_time_subsample"] = min(cfg.get("ig_time_subsample", 8), 6)

                cfg["capture_training_curves"] = True

                results = run_analysis(algo, df, cfg)
                st.session_state["results"] = results
                st.success("✅ Training completed with hazard playback enabled.")
            except Exception as e:
                st.error(f"Run failed: {e}")

    if run_clicked:
        with st.spinner(f"Training {algo}..."):
            try:
                cfg = dict(config)
                results = run_analysis(algo, df, cfg)
                st.session_state["results"] = results
                st.success("✅ Training completed.")
            except Exception as e:
                st.error(f"Run failed: {e}")

    # ===================== 6) Results =========================================
    if "results" in st.session_state:
        results = st.session_state["results"]
        _render_metrics_block(results)

        curve_data = results.get("training_curve_history")
        if isinstance(curve_data, dict) and (curve_data.get("entries")):
            st.subheader("🎬 Hazard / Survival Training Playback")
            _render_training_curve_history(curve_data)
        elif algo == "TEXGISA":
            st.info("Run **Train + Hazard Playback** to capture per-epoch hazards and survival curves for this widget.")

        # Survival curves
        st.subheader("Charts")
        chart_opt = st.selectbox(
            "Select a chart to display",
            ["Predicted Survival Trajectories", "Kaplan–Meier (overall)", "Kaplan–Meier by risk groups"],
            help="Choose which diagnostic chart to render from the latest training run.",
        )

        # Configure Kaplan–Meier options when a KM chart is selected.
        limit_to_last_event = False
        km_bin_width = 0
        if chart_opt.startswith("Kaplan–Meier"):
            c1, c2 = st.columns(2)
            with c1:
                limit_to_last_event = st.checkbox(
                    "Limit x-axis to last event time",
                    value=True,
                    help="Trim the Kaplan–Meier curve after the final observed event to focus on informative intervals.",
                )
            with c2:
                km_bin_width = st.number_input("KM bin width (0 = none)", min_value=0, value=0, step=1,
                                            help="Aggregate time into fixed-width bins (e.g., by month or year); 0 indicates no binning.")

        if chart_opt == "Predicted Survival Trajectories":
            if isinstance(results.get("Surv_Test"), pd.DataFrame):
                _plot_survival_curves(results["Surv_Test"])
            else:
                st.info("No predicted survival table available yet.")
        elif chart_opt == "Kaplan–Meier (overall)":
            _plot_km_overall(df, limit_to_last_event=limit_to_last_event, bin_width=km_bin_width)
        elif chart_opt == "Kaplan–Meier by risk groups":
            _plot_km_by_risk(df, results.get("Surv_Test"), n_groups=3,
                            limit_to_last_event=limit_to_last_event, bin_width=km_bin_width)


        # Feature importance
        if isinstance(results.get("Feature Importance"), pd.DataFrame):
            # === Feature Importance (TEXGISA) — visualisation and downloads ===
            st.subheader("Feature Importance (TEXGISA)")

            # Extract the feature-importance table from results regardless of its original key or structure.
            fi_df = _extract_fi_df(results)

            if fi_df is None or fi_df.empty:
                st.info("No feature importance is available in the results.")
            else:
                # Plot the top-k (or all) rows.
                _render_fi_plot(fi_df, topn=10)

                # Collapsible table that exposes the full data for inspection.
                with st.expander("See all features", expanded=False):
                    try:
                        st.dataframe(fi_df, use_container_width=True, hide_index=True)
                    except TypeError:
                        st.dataframe(fi_df, use_container_width=True)

                # Download helpers for the complete table and single columns.
                c1, c2 = st.columns([0.55, 0.45])

                # Full CSV export.
                csv_full = fi_df.to_csv(index=False).encode("utf-8")
                c1.download_button(
                    "📥 Download full FI (CSV)",
                    data=csv_full,
                    file_name="feature_importance_full.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # Export an individual metric column (importance or directional mean).
                metric_choices = [c for c in ["importance", "directional_mean"] if c in fi_df.columns]
                if metric_choices:
                    with c2:
                        metric = st.selectbox(
                            "Metric to export",
                            metric_choices,
                            index=0,
                            key="fi_metric_export",
                            help="Choose which metric column to export as a standalone CSV.",
                        )
                        csv_metric = fi_df[["feature", metric]].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"📥 Download {metric} only (CSV)",
                            data=csv_metric,
                            file_name=f"feature_{metric}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            # Preserve any additional TEXGISA downloads (for example time-dependent tensors) if available.
            if "texgi_time_tensor_path" in results:
                st.download_button(
                    "📥 Download time-dependent TEXGISA (pt)",
                    data=open(results["texgi_time_tensor_path"], "rb").read(),
                    file_name="texgi_time_tensor.pt",
                    mime="application/octet-stream",
                )


        # Raw per-time attributions file
        if "Phi_Val_Path" in results:
            try:
                with open(results["Phi_Val_Path"], "rb") as f:
                    st.download_button("⬇️ Download time-dependent TEXGISA (pt)", f, file_name="mysa_phi_val.pt", type="secondary")
            except Exception:
                pass


def run_analysis(algo: str, df: pd.DataFrame, config: dict):
    algo = algo.lower()
    # Route to underlying algorithms
    if algo.startswith("coxtime"):
        return coxtime.run_coxtime(df, config)
    elif algo.startswith("deepsurv"):
        return deepsurv.run_deepsurv(df, config)
    elif algo.startswith("deephit"):
        return deephit.run_deephit(df, config)
    elif "texgisa" in algo or "mysa" in algo:
        if run_texgisa is None:
            raise RuntimeError("TEXGISA not available. Please ensure models/mysa.py is present.")
        cfg = dict(config)
        if "multimodal_sources" not in cfg:
            mm_sources = _build_multimodal_sources(cfg)
            if mm_sources is not None:
                cfg["multimodal_sources"] = mm_sources
        return run_texgisa(df, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
