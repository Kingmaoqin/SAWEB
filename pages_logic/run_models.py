
# run_models.py — MySA-integrated Workbench (drop-in replacement)
# This page supports: CoxTime, DeepSurv, DeepHit, and MySA (TEXGI).
# - Adds two loss-weight sliders (lambda_smooth, lambda_expert)
# - Adds an editable Expert Rules table (relation/sign/min_mag/weight)
# - Shows TEXGI Feature Importance (Top-K + expand to all), and allows downloading time-dependent attributions.

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from models import coxtime, deepsurv, deephit
from models.mysa import run_mysa as run_texgisa
def _md_explain(text: str, size: str = "1.12rem", line_height: float = 1.6):
    """把解释文字放大显示。size 可改为 1.2rem/1.3rem 等。"""
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
            return "This bar chart ranks features by TEXGI importance. Larger bars indicate stronger contribution to the predicted risk over time."

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
            f"This bar chart displays the top-{k} features ranked by TEXGI importance. "
            f"The most influential feature is {top_feat} (importance {top_imp:.4f}). "
            f"On average, the importance across the shown features is {avg_imp:.4f}. "
        )
        if has_dir:
            # 颜色与你的绘图一致：正=蓝(#60a5fa)，负=红(#f87171)
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
    """Maintain the help text (Markdown) for each parameter centrally. Add/modify as needed."""
    HELP = {
        # Data Mapping
        "time_col":       "The column in the dataset representing follow-up time/survival time. Must be numeric; units are self-defined (e.g., days/months/years).",
        "event_col":      "The event indicator column: 1=event occurred (e.g., death/recurrence), 0=censored (did not occur before the end of the study).",
        "features":       "The feature columns to be used for modeling (excluding time and event columns).",
        "positive_label": "Maps samples in the event column equal to this value to 1 (event occurred), and all others to 0 (censored).",

        # Algorithm Selection & Common Training Parameters
        "algo":           "Select the training algorithm. TEXGISA supports time-dependent explanations and expert priors; CoxTime/DeepSurv/DeepHit are common baselines.",
        "batch_size":     "The number of samples used for each parameter update. Can be reduced if GPU memory is tight; more stable but slower.",
        "epochs":         "The number of training epochs. A larger value is generally more stable but takes longer. It's recommended to start with 50~150 to observe convergence.",
        "lr":             "Learning rate. Too large can cause oscillations, too small makes training very slow. You can start with a range of 1e-3 ~ 1e-2.",
        "val_split":      "The proportion of training data to be used for validation, for early stopping and best epoch selection.",

        # DeepHit
        "num_intervals":  "The number of intervals to discretize continuous time into (used only by DeepHit/discrete-time models). Too many can lead to sparsity, too few can be too coarse.",

        # MySA Regularization & Priors
        "lambda_expert":  "The weight for the expert prior penalty (λ_expert). A larger value enforces stronger adherence to expert rules/importance; too large may sacrifice predictive performance.",
        "lambda_smooth":  "The weight for smoothness in the time dimension (λ_smooth). Makes explanations smoother across adjacent time points; too large may mask true time-dependent effects.",
        "fast_mode":      "Acceleration mode: Uses a lightweight generator and approximate TEXGI for a quick preview of the expert prior's effect. Results may differ slightly from the full version.",
        "ig_steps":       "The number of integration steps for calculating Integrated Gradients (TEXGI). Larger is more accurate but slower (commonly 16~64).",

        # Generator / TEXGI Advanced Parameters
        "latent_dim":       "The dimension of the generator's noise vector (latent variable).",
        "extreme_dim":      "The dimension of the extreme encoding vector (for modeling extreme risk directions).",
        "gen_epochs":       "The number of training epochs for the generator (used only in TEXGI).",
        "gen_batch":        "The batch size for generator training.",
        "gen_lr":           "The learning rate for generator optimization.",
        "gen_alpha_dist":   "The generator's distribution distance regularization weight α (a larger value means closer to the reference distribution).",
        "ig_batch_samples": "The number of samples B' to draw per batch in TEXGI (larger is more stable but slower).",
        "ig_time_subsample":"The number of time steps T' to sample each time in TEXGI (subsampling time to accelerate).",
    }
    return HELP.get(key, "No description available (you can add a description for this key in _qhelp_md)")

def field_with_help(control_fn, label, help_key: str, *args, **kwargs):
    """
    优先用 Streamlit 原生 help=（控件标签右侧会出现 (i) 图标）；
    老版本没有 help= 时，右侧放一个 ❔，点击弹出说明/或在下方显示。
    用法不变：epochs = field_with_help(st.number_input, "Epochs", "epochs", 10, 2000, 150, step=10)
    """
    help_msg = _qhelp_md(help_key)

    # 1) 最稳：多数 st.* 控件都支持 help= 参数（官方 (i) icon）
    try:
        return control_fn(label, *args, help=help_msg, **kwargs)
    except TypeError:
        # 2) 兜底：没有 help= 的控件，用右侧 ❔（popover→button）
        c1, c2 = st.columns([0.96, 0.04])  # 右侧给足空间，避免被吃掉
        with c1:
            val = control_fn(label, *args, **kwargs)
        with c2:
            try:
                # 新版 popover
                with st.popover("❔"):
                    st.markdown(help_msg)
            except Exception:
                # 老版：按钮点击在控件下方显示说明
                if st.button("❔", key=f"help_{help_key}"):
                    st.info(help_msg)
        return val


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

        # --- 关键修复：列名统一转为字符串再做匹配 ---
        cols_lower = [str(c).lower() for c in df.columns]

        # 如果 feature 不在列里，且索引有名字/多级名，则先提到列
        has_index_name = (getattr(df.index, "name", None) is not None) or (
            hasattr(df.index, "names") and any(n is not None for n in (df.index.names or []))
        )
        if "feature" not in cols_lower and has_index_name:
            df = df.reset_index()
            cols_lower = [str(c).lower() for c in df.columns]

        # 构造 “小写字符串列名 -> 原列名” 的映射
        colmap = {str(c).lower(): c for c in df.columns}

        # 1) feature 列
        feat_col = None
        for cand in ("feature", "index", "name", "variable", "feat", "feature_name"):
            if cand in colmap:
                feat_col = colmap[cand]
                break
        if feat_col is None:
            # 若无显式列，尝试找一个对象列当作 feature
            obj_cols = [c for c in df.columns if df[c].dtype == "O"]
            if obj_cols:
                feat_col = obj_cols[0]
            else:
                # 再兜底：如果第一列看起来像类别/字符串，就用它
                if len(df.columns) >= 1 and df[df.columns[0]].dtype == "O":
                    feat_col = df.columns[0]
                else:
                    return None

        # 2) importance 列（考虑别名）
        imp_col = None
        for cand in ("importance", "score", "weight", "texgi", "attr", "phi", "value"):
            if cand in colmap:
                imp_col = colmap[cand]
                break
        if imp_col is None:
            # 随机取一个数值列（但不能是 feature 列）
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != feat_col]
            if num_cols:
                imp_col = num_cols[0]
            else:
                return None

        # 3) 方向列（可选）
        dir_col = None
        for cand in ("directional_mean", "direction", "signed_mean", "signed"):
            if cand in colmap:
                dir_col = colmap[cand]
                break

        # 统一重命名
        rename_map = {feat_col: "feature", imp_col: "importance"}
        if dir_col:
            rename_map[dir_col] = "directional_mean"
        df = df.rename(columns=rename_map)

        # 只保留关键列，并规范类型
        keep = ["feature", "importance"] + (["directional_mean"] if "directional_mean" in df.columns else [])
        df = df[keep]
        df["feature"] = df["feature"].astype(str)
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        if "directional_mean" in df.columns:
            df["directional_mean"] = pd.to_numeric(df["directional_mean"], errors="coerce")
        df = df.dropna(subset=["importance"])
        return df

    # 1) 先按常见键名取
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

    # 2) 兜底：在所有 values 里找 “像 FI 的 DataFrame”
    for v in results.values():
        if isinstance(v, pd.DataFrame):
            df = _normalize_df(v)
            if df is not None:
                return df

    return None


def _render_fi_plot(fi_df: pd.DataFrame, topn: int = 10):
    """Plot TEXGI Top-k as a horizontal bar chart (k<=10)."""
    if fi_df is None or fi_df.empty:
        st.info("No feature importance available.")
        return

    df = fi_df.copy()
    # 重要性列名按你当前表头：importance / directional_mean
    if "importance" in df.columns:
        df = df.sort_values("importance", ascending=False)
    # 只取前 10（不足 10 就全量）
    k = min(topn, len(df))
    df_top = df.head(k)

    # 类别与数值
    y_labels = list(reversed(df_top["feature"].astype(str).tolist()))
    x_vals   = list(reversed(df_top["importance"].astype(float).tolist()))

    # 如果有方向信息，用颜色区分：正（蓝）/ 负（红）
    colors = None
    if "directional_mean" in df_top.columns:
        signs = df_top["directional_mean"].apply(lambda v: 1 if v >= 0 else -1).tolist()
        colors = ["#60a5fa" if s > 0 else "#f87171" for s in reversed(signs)]

    # 画图
    import matplotlib.pyplot as plt
    fig_h = 4 + 0.35 * k  # 根据条数自适应高度
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(y_labels, x_vals, color=colors)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(f"TEXGI Top-{k} Features")
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
    # 组装所有数值型指标
    items = []
    for k, v in results.items():
        if isinstance(v, (int, float, np.floating)):
            items.append((str(k), float(v)))
    if not items:
        return

    _inject_metrics_css()


    # 规则：把一些常见指标漂亮展示；其余自动识别
    def _fmt_val(name, val):
        if "index" in name.lower() or "c-index" in name.lower() or "cindex" in name.lower():
            return f"{val:.4f}"
        if name.lower().endswith(("epoch", "epochs", "samples", "bins", "events")) or abs(val - round(val)) < 1e-9:
            return f"{int(round(val)):,}"
        return f"{val:.4f}"

    # 渲染 KPI 卡片
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

    # 如果需要，也保留一个“展开表格”版本
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
            # 兼容更旧的 Streamlit
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


def _build_expert_rules_from_editor(df_rules):
    """Convert data_editor dataframe to expert_rules dict expected by MySA."""
    rules = []
    if df_rules is None or df_rules.empty:
        return {"rules": []}
    for _, row in df_rules.iterrows():
        feat = str(row.get("feature","")).strip()
        if not feat:
            continue
        relation = row.get("relation", None)
        if relation in ("none", "", None):
            relation = None
        sign = int(row.get("sign", 0)) if pd.notna(row.get("sign", 0)) else 0
        min_mag = float(row.get("min_mag", 0.0)) if pd.notna(row.get("min_mag", 0.0)) else 0.0
        weight = float(row.get("weight", 1.0)) if pd.notna(row.get("weight", 1.0)) else 1.0
        rule = {"feature": feat, "weight": weight}
        if relation is not None:
            rule["relation"] = relation
        if sign != 0:
            rule["sign"] = sign
        if min_mag > 0.0:
            rule["min_mag"] = min_mag
        rules.append(rule)
    return {"rules": rules}


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

def _compute_km(durations, events, limit_to_last_event=True, bin_width=0):
    """
    durations: 1D array-like of times (float/int)
    events   : 1D array-like of {0,1}
    limit_to_last_event: True -> 仅用到最后一个事件时间（之后全是删失，KM 不再下降）
    bin_width: >0 时，将时间按此宽度分箱（如按月/年），用箱右端点作为时间坐标
    返回 (t, S)，其中 t 仅包含“发生事件的时刻”的节点，使阶梯更明显
    """
    import numpy as np
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)

    if bin_width and bin_width > 0:
        # 把时间映射到 bin 的右端点（更直观）
        d = (np.floor(d / bin_width) * bin_width).astype(float)

    if limit_to_last_event and (e == 1).any():
        t_last_event = d[e == 1].max()
        keep = d <= t_last_event
        d, e = d[keep], e[keep]

    # 按时间排序
    order = np.argsort(d)
    d, e = d[order], e[order]

    uniq = np.unique(d)
    n_at_risk = len(d)
    S = 1.0
    curve_t = [0.0]
    curve_S = [1.0]

    # 只在“发生事件的时刻”更新 S；纯删失时仅减少风险集，不更新 S
    for tt in uniq:
        at_t = (d == tt)
        m = at_t.sum()                # 在 tt 的总个体数（事件+删失）
        d_events = (e[at_t] == 1).sum()  # 在 tt 的事件数
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

    # 对齐索引（与之前一致）
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
    )

    # === New: Image ZIP wizard (ResNet-50 -> Features) =============================
    with st.expander("🖼️ Build Training Data from Image Data (ResNet-50 → Features)", expanded=False):
        st.markdown(
            "Steps: Upload **Image ZIP** → Complete/Import **Survival Labels** → One-click generate data table and directly train MySA.\n\n"
            "**Label Definitions**: `duration` is the follow-up time, `event` indicates if the event occurred (0=No, 1=Yes)."
        )

        # 1) Upload ZIP (Required)
        zip_up = st.file_uploader("① Upload Image ZIP (Required)", type=["zip"], key="img_zip_simple")

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
                labels_up = st.file_uploader("Or upload your completed label CSV", type=["csv"], key="img_labels_csv")

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
            st.subheader("③ Generate Training Table (and optionally run a quick test with MySA)")
            c3, c4, c5 = st.columns(3)
            with c3:
                backbone = st.selectbox("Feature Backbone Network", ["resnet50"], index=0, help="Can be extended to ViT/CLIP after confirming it runs")
            with c4:
                img_bs = st.number_input("Feature Extraction Batch Size", 8, 256, 64, step=8)
            with c5:
                img_workers = st.number_input("DataLoader Parallel Workers", 0, 8, 2, step=1)

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
                    st.dataframe(df_img.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed: {e}")

            # Optional: One-click quick run with MySA (small epochs for non-technical user experience)
            try:
                from models.mysa import run_mysa as _quick_mysa
            except Exception:
                _quick_mysa = None

            if _quick_mysa is not None and st.session_state.get("clinical_data") is not None:
                if st.button("🚀 One-Click Test Run with MySA (Quick Metrics Preview)", use_container_width=True):
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
                    with st.spinner("Training MySA (Quick Preview)..."):
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
    )

    with st.expander("📈 Build Training Data from Sensor Data (Full-sequence Features)", expanded=False):
        st.markdown(
            "Steps: Upload **Sensor ZIP** (one file per sample, CSV/Parquet) → Complete/Import `file,duration,event` → "
            "Extract **full-sequence features** at once → Generate training table and directly train MySA.\n\n"
            "**Note**: By default, statistical and frequency-domain features are extracted from the entire sequence without a sliding window. If files contain timestamps, you can select a resampling frequency."
        )

        zip_up_sens = st.file_uploader("① Upload Sensor ZIP (Required)", type=["zip"], key="sensor_zip")
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
                labels_up = st.file_uploader("Upload Your Completed Label CSV", type=["csv"], key="sensor_labels_csv")

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
            use_only_labeled = st.checkbox("Use only samples with completed labels", value=True)
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
                    st.dataframe(df_sens.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed: {e}")
    # === End of sensor wizard ======================================================


    # ===================== Multimodal data upload =============================
    with st.expander("🗂 Multimodal Data Upload", expanded=False):
        tab_up = st.file_uploader("Tabular CSV", type=["csv"], key="mm_tabular")
        img_up = st.file_uploader("Image CSV", type=["csv"], key="mm_image")
        sens_up = st.file_uploader("Sensor CSV", type=["csv"], key="mm_sensor")

        if tab_up is not None:
            tab_df = pd.read_csv(tab_up)
            st.session_state["mm_tabular_df"] = tab_df
            st.dataframe(tab_df.head(), use_container_width=True)
        if img_up is not None:
            img_df = pd.read_csv(img_up)
            st.session_state["mm_image_df"] = img_df
            st.dataframe(img_df.head(), use_container_width=True)
        if sens_up is not None:
            sens_df = pd.read_csv(sens_up)
            st.session_state["mm_sensor_df"] = sens_df
            st.dataframe(sens_df.head(), use_container_width=True)

        has_any = any(
            k in st.session_state for k in ["mm_tabular_df", "mm_image_df", "mm_sensor_df"]
        )
        if has_any:
            st.markdown("### Field Alignment")
            tab_id = img_id = sens_id = None
            if "mm_tabular_df" in st.session_state:
                tab_id = st.selectbox(
                    "Tabular ID column",
                    st.session_state["mm_tabular_df"].columns,
                    key="mm_tab_id",
                )
            if "mm_image_df" in st.session_state:
                img_id = st.selectbox(
                    "Image ID column",
                    st.session_state["mm_image_df"].columns,
                    key="mm_img_id",
                )
            if "mm_sensor_df" in st.session_state:
                sens_id = st.selectbox(
                    "Sensor ID column",
                    st.session_state["mm_sensor_df"].columns,
                    key="mm_sens_id",
                )

            if st.button("Load Multimodal Data"):
                combined = st.session_state.get("mm_tabular_df")
                if combined is None:
                    st.warning("Tabular data is required for alignment.")
                else:
                    if "mm_image_df" in st.session_state and img_id:
                        combined = combined.merge(
                            st.session_state["mm_image_df"],
                            left_on=tab_id,
                            right_on=img_id,
                            how="left",
                        )
                    if "mm_sensor_df" in st.session_state and sens_id:
                        combined = combined.merge(
                            st.session_state["mm_sensor_df"],
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
                    st.dataframe(combined.head(), use_container_width=True)

    # ===================== 1) Data upload & preview ===========================
    with st.expander("📘 Step-by-Step Guide for Tabular Data", expanded=False):
        st.markdown(
            "- Upload a CSV file with **duration** and **event** columns, plus features.\n"
            "- Select columns and model, set hyperparameters, and run.\n"
            "- For **MySA (TEXGI)**, you can set **λ_smooth** and **λ_expert**, and edit **Expert Rules**."
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
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


    # Build working dataframe with required names
    try:
        # 先取用户选择的列并规范列名
        base = data[features + [time_col, event_col]].copy()
        df = base.rename(columns={time_col: "duration", event_col: "event"})

        # 事件列映射为 0/1（保持你原有的正类映射逻辑）
        df["event"] = _ensure_binary_event(
            df["event"],
            positive=type(df["event"].iloc[0])(event_positive)
        )

        # ========== 🧹 新增：自动清洗 UI 开关 & 参数 ==========
        st.markdown("### 🧹 Auto-clean Data")
        c0, c1, c2 = st.columns([0.45, 0.35, 0.20])
        with c0:
            auto_clean = st.checkbox(
                "Enable auto-clean (bool→0/1, drop high-NaN/constant, Z-score)",
                value=True, key="csv_autoclean"
            )
        with c1:
            nan_thresh = st.slider(
                "Missing-value threshold to drop a column",
                min_value=0.0, max_value=0.9, value=0.30, step=0.05,
                key="csv_nan_thresh"
            )
        with c2:
            do_zscore = st.checkbox("Z-score", value=True, key="csv_zscore")

        # 工具函数（局部定义，避免污染全局）
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

        # ========== 应用清洗（仅作用于用户选中的 features） ==========
        if auto_clean:
            # 复制特征子表
            X = df[features].copy()

            # 1) 布尔/字符串布尔 → 0/1；其它非数值尝试转数值
            for f in list(X.columns):
                s = X[f]
                if _is_bool_like(s):
                    X[f] = _to_bool01(s)
                elif not np.issubdtype(s.dtype, np.number):
                    X[f] = _pd.to_numeric(s, errors="coerce")

            # 2) 仅保留数值特征
            X = X.select_dtypes(include=["number"])

            # 3) 丢弃高缺失/零方差/常数列
            nan_ratio = X.isna().mean()
            std = X.std(ddof=0)
            keep_mask = (nan_ratio <= float(nan_thresh)) & (std > 1e-12)
            dropped_nan = int((nan_ratio > float(nan_thresh)).sum())
            dropped_const = int((std <= 1e-12).sum())
            X = X.loc[:, keep_mask].fillna(0.0)

            # 4) 可选 Z-score（全局；验证有效后可迁到 trainer 用 train-only 统计）
            if do_zscore and X.shape[1] > 0:
                mu = X.mean(0)
                sigma = X.std(0).replace(0.0, 1.0)
                X = (X - mu) / sigma

            # 5) 组装回最终训练表（duration,event + 纯数值特征）
            df = _pd.concat([df[["duration","event"]], X.astype("float32")], axis=1)

            # 同步“特征列列表”供后续 config 使用
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
            # 不清洗：至少把非数值特征做基本 to_numeric（保底）
            for f in features:
                if not np.issubdtype(df[f].dtype, np.number):
                    df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)

    except Exception as e:
        st.error(f"Column mapping/cleaning failed: {e}")
        return


    # ===================== 3) Algorithm & hyperparameters =====================
    st.subheader("⚙️ Algorithm & Training Configuration")
    algo = field_with_help(
        st.selectbox, "Algorithm", "algo",
        ["CoxTime", "DeepSurv", "DeepHit", "TEXGISA (MySA)"]
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


    # ===================== 4) MySA regularizers & expert rules ================
    if algo == "TEXGISA (MySA)":
        st.markdown("### Regularizers")
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

        st.markdown("### Expert Rules")
        # Prepare blank editor with choices
        options_relation = ["none", ">=mean", "<=mean"]
        options_sign = [-1, 0, +1]

        # Seed table with Top-10 placeholders (feature dropdowns)
        seed = pd.DataFrame({
            "feature": (features + [""]*10)[:10],
            "relation": ["none"]*10,
            "sign": [0]*10,
            "min_mag": [0.0]*10,
            "weight": [1.0]*10,
        })
        edited = st.data_editor(
            seed,
            column_config={
                "feature": st.column_config.SelectboxColumn("feature", options=features, help="Choose a feature"),
                "relation": st.column_config.SelectboxColumn("relation", options=options_relation),
                "sign": st.column_config.SelectboxColumn("sign", options=options_sign),
                "min_mag": st.column_config.NumberColumn("min_mag", step=0.01, format="%.3f"),
                "weight": st.column_config.NumberColumn("weight", step=0.1, format="%.2f"),
            },
            num_rows="dynamic",
            use_container_width=True
        )
        expert_rules = _build_expert_rules_from_editor(edited)

        # Advanced TEXGI/Generator controls
        with st.expander("Advanced TEXGI / Generator settings", expanded=False):
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
                st.number_input, "Time-bins per batch for TEXGI (T')", "ig_time_subsample",
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

    # ===================== 5) Run =============================================
    c_run1, c_run2 = st.columns(2)
    with c_run1:
        preview_clicked = st.button("👀 Preview FI (no expert priors)", use_container_width=True)
    with c_run2:
        train_clicked = st.button("🚀 Train with Expert Priors", use_container_width=True)
        fast_expert = st.checkbox("Fast expert mode (lighter generator & TEXGI)", value=True)


    if preview_clicked:
        with st.spinner("Training briefly and computing attributions (λ_expert=0)..."):
            try:
                cfg = dict(config)
                cfg["lambda_expert"] = 0.0              # 预览时不加专家惩罚
                # 也给个较小的预览 epoch，减少等待（你可以改成想要的值）
                cfg["epochs"] = min(int(cfg.get("epochs", 200)), 50)
                results = run_analysis(algo, df, cfg)
                st.session_state["results"] = results
                st.success("✅ FI preview done.")
            except Exception as e:
                st.error(f"Preview failed: {e}")

    # if train_clicked:
    #     with st.spinner("Training with expert priors..."):
    #         try:
    #             results = run_analysis(algo, df, config)  # 使用当前界面设置，包括 λ_expert 和专家规则
    #             st.session_state["results"] = results
    #             st.success("✅ Training completed.")
    #         except Exception as e:
    #             st.error(f"Run failed: {e}")
    if train_clicked:
        with st.spinner("Training with expert priors..."):
            try:
                # 创建配置的浅拷贝以进行潜在的修改
                cfg = dict(config)

                # 如果 fast_expert 为 True，则调整配置参数
                if fast_expert:
                    cfg["gen_epochs"] = min(cfg.get("gen_epochs", 200), 40)
                    cfg["ig_steps"] = min(cfg.get("ig_steps", 20), 10)
                    cfg["ig_batch_samples"] = min(cfg.get("ig_batch_samples", 32), 24)
                    cfg["ig_time_subsample"] = min(cfg.get("ig_time_subsample", 8), 6)
                
                # 使用可能已调整的配置（cfg）运行分析
                results = run_analysis(algo, df, cfg) 
                
                st.session_state["results"] = results
                st.success("✅ Training completed.")
                
            except Exception as e:
                st.error(f"Run failed: {e}")

    # ===================== 6) Results =========================================
    if "results" in st.session_state:
        results = st.session_state["results"]
        _render_metrics_block(results)

        # Survival curves
        st.subheader("Charts")
        chart_opt = st.selectbox(
            "Select a chart to display",
            ["Predicted Survival Trajectories", "Kaplan–Meier (overall)", "Kaplan–Meier by risk groups"]
        )

        # KM 选项（仅在 KM 时显示）
        limit_to_last_event = False
        km_bin_width = 0
        if chart_opt.startswith("Kaplan–Meier"):
            c1, c2 = st.columns(2)
            with c1:
                limit_to_last_event = st.checkbox("Limit x-axis to last event time", value=True)
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
            # === Feature Importance (TEXGI) — 图形化展示 + 下载 ===
            st.subheader("Feature Importance (TEXGI)")

            # 从 results 中鲁棒提取 FI 表（自动识别 fi_table / texgi_importance 等多种命名与结构）
            fi_df = _extract_fi_df(results)

            if fi_df is None or fi_df.empty:
                st.info("No feature importance is available in the results.")
            else:
                # 前 10（不足 10 就全量）绘图
                _render_fi_plot(fi_df, topn=10)

                # “查看全部”折叠表（保留原始信息，便于检查/复制）
                with st.expander("See all features", expanded=False):
                    try:
                        st.dataframe(fi_df, use_container_width=True, hide_index=True)
                    except TypeError:
                        st.dataframe(fi_df, use_container_width=True)

                # 下载区：完整列表 + 单指标
                c1, c2 = st.columns([0.55, 0.45])

                # 完整 CSV
                csv_full = fi_df.to_csv(index=False).encode("utf-8")
                c1.download_button(
                    "📥 Download full FI (CSV)",
                    data=csv_full,
                    file_name="feature_importance_full.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # 选择某个指标单独导出（importance / directional_mean）
                metric_choices = [c for c in ["importance", "directional_mean"] if c in fi_df.columns]
                if metric_choices:
                    with c2:
                        metric = st.selectbox("Metric to export", metric_choices, index=0, key="fi_metric_export")
                        csv_metric = fi_df[["feature", metric]].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"📥 Download {metric} only (CSV)",
                            data=csv_metric,
                            file_name=f"feature_{metric}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            # 若你的结果里还有“时间依赖 TEXGI（pt 张量）”的下载，保留原按钮
            if "texgi_time_tensor_path" in results:
                st.download_button(
                    "📥 Download time-dependent TEXGI (pt)",
                    data=open(results["texgi_time_tensor_path"], "rb").read(),
                    file_name="texgi_time_tensor.pt",
                    mime="application/octet-stream",
                )


        # Raw per-time attributions file
        if "Phi_Val_Path" in results:
            try:
                with open(results["Phi_Val_Path"], "rb") as f:
                    st.download_button("⬇️ Download time-dependent TEXGI (pt)", f, file_name="mysa_phi_val.pt", type="secondary")
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
            raise RuntimeError("TEXGISA/MySA not available. Please ensure models/mysa.py is present.")
        return run_texgisa(df, config)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
