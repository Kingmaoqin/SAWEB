# pages_logic/chat_with_agent.py — upload + real tool calls (agent & direct), clean UI

from __future__ import annotations

import json
import re
from typing import List, Optional, Dict, Any

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# project agent & tools
from sa_agent import sa_agent
from sa_tools import get_data_summary, run_survival_analysis

ALGORITHM_GUIDE: Dict[str, Dict[str, str]] = {
    "TEXGISA": {
        "summary": "Multimodal survival analysis with TEXGISA explanations and optional expert priors.",
        "best_for": "multimodal data, expert priors, attribution reporting",
    },
    "CoxTime": {
        "summary": "Neural Cox model that captures time-varying effects.",
        "best_for": "dynamic hazards over time",
    },
    "DeepSurv": {
        "summary": "Deep learning extension of Cox PH for non-linear tabular relationships.",
        "best_for": "non-linear risk patterns",
    },
    "DeepHit": {
        "summary": "Neural model designed for competing risks scenarios.",
        "best_for": "multiple event types",
    },
}

# --- dataset state helpers (add near the top) ---
def _dataset_state():
    dm = st.session_state.get("data_manager")
    if dm is None:
        return {"loaded": False, "name": None, "cols": []}
    summ = dm.get_data_summary() or {}
    loaded = "error" not in summ
    name = getattr(dm, "current_file_name", None) or summ.get("file_name")
    cols = summ.get("column_names", []) or []
    return {"loaded": loaded, "name": name, "cols": cols}

def _context_string():
    s = _dataset_state()
    algo_lines = [
        f"- {name}: {info['summary']} (best for {info['best_for']})"
        for name, info in ALGORITHM_GUIDE.items()
    ]
    return (
        "[STATE]\n"
        f"DATASET_LOADED={s['loaded']}\n"
        f"DATASET_NAME={s['name'] or 'None'}\n"
        f"COLUMNS={','.join(map(str, s['cols']))}\n"
        "RULES:\n"
        "- If DATASET_LOADED is False, you MUST say the dataset is not loaded and ask the user to upload a CSV on the right panel.\n"
        "- Never claim to have inspected or loaded data unless DATASET_LOADED is True.\n"
        "- When the user requests an algorithm, select from TEXGISA, CoxTime, DeepSurv, or DeepHit and confirm the time/event columns.\n"
        "- Before executing training or preview runs, explicitly ask the user to reply 'yes' or 'no'. Only proceed after they answer.\n"
        "- Always confirm which columns represent time and event; propose guesses from COLUMNS and let the user override.\n"
        "- Prefer the preview/train/run shortcuts when the user wants quick execution; otherwise ask for the time/event columns explicitly.\n"
        "ALGORITHMS:\n"
        + "\n".join(algo_lines)
    )

# ------------------------ helpers ------------------------

def _ensure_data_manager():
    """Make sure session has a shared DataManager instance."""
    if "data_manager" not in st.session_state:
        from sa_data_manager import get_shared_manager

        st.session_state.data_manager = get_shared_manager()


def _queue_confirmation(label: str, fn, kwargs: dict):
    """Store a pending action that requires an explicit yes/no before execution."""
    st.session_state["pending_action"] = {
        "label": label,
        "fn": fn,
        "kwargs": kwargs,
    }


def _queue_and_prompt(label: str, fn, kwargs: dict, prompt: str, *, rerun: bool = False):
    """Queue a confirmation request and immediately surface the prompt.

    When rerun=True, the page reruns right away so the new assistant message and
    pending warning are visible without needing additional user input.
    """
    _queue_confirmation(label, fn, kwargs)
    st.session_state.chat_messages.append(AIMessage(content=prompt))
    if rerun:
        st.rerun()


def _run_pending_if_confirmed(response_text: str) -> bool:
    """Handle a yes/no response for the queued action.

    Returns True if the response was a valid confirmation (yes or no), False otherwise.
    """
    pending = st.session_state.get("pending_action")
    if not pending:
        return False

    answer = response_text.strip().lower()
    if answer not in {"yes", "y", "no", "n"}:
        # Ask the user to answer explicitly.
        with st.chat_message("assistant"):
            st.markdown("Please reply with **yes** or **no** to continue the pending action.")
        st.session_state.chat_messages.append(AIMessage(content="Please reply yes or no to continue."))
        return True

    if answer in {"no", "n"}:
        msg = f"Canceled: {pending['label']}"
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.chat_messages.append(AIMessage(content=msg))
        st.session_state.pop("pending_action", None)
        return True

    # User confirmed
    with st.chat_message("assistant"):
        st.markdown(f"Running: {pending['label']}")
        with st.spinner("Executing requested analysis..."):
            res = pending["fn"](**pending["kwargs"])
    st.session_state.chat_messages.append(AIMessage(content=f"Started: {pending['label']}"))
    st.session_state.pop("pending_action", None)
    # Preserve last results for the result panel when _run_direct returns a dict
    if isinstance(res, dict):
        st.session_state["last_results"] = res
    return True


def _style():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .side-card { 
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px; 
            padding: 0.9rem 1.0rem; 
            background: rgba(255,255,255,0.03);
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .chip-row .stButton {
            flex: 1 1 200px;
        }
        .chip-row .stButton>button {
            border-radius: 999px;
            padding: 0.42rem 0.9rem;
            font-size: 0.9rem;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
        }
        .chip-row .stButton>button:hover { background: rgba(255,255,255,0.08); }
        .stChatMessage { margin-bottom: 0.6rem; }
        .result-card {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            background: rgba(255,255,255,0.02);
            margin-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_history(messages: List):
    """Render chat history (display only; do NOT append here)."""
    for m in messages:
        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.markdown(m.content)
        elif isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(m.content)


def _guess_cols_from_summary(summary: dict) -> tuple[Optional[str], Optional[str]]:
    """Heuristic guess of (time_col, event_col) from data summary."""
    cols = summary.get("column_names", []) or []
    time_col = None
    event_col = None
    for c in cols:
        lc = c.lower()
        if time_col is None and re.search(r"(time|duration|followup|surv|t_?)$", lc):
            time_col = c
        if event_col is None and re.search(r"(event|status|death|dead|outcome)$", lc):
            event_col = c
    if time_col is None and "duration" in cols: time_col = "duration"
    if event_col is None and "event" in cols: event_col = "event"
    return time_col, event_col


def _inject_user(text: str):
    """Push a piece of text as if the user typed it, then rerun once."""
    st.session_state["__inject_user"] = text
    st.rerun()


def _process_user_text(user_text: str):
    """Append user message, then call agent ONCE with a prefixed context message."""
    st.session_state.chat_messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # prepend a context message (NOT added to history)
    ctx = AIMessage(content=_context_string())
    history = st.session_state.chat_messages[:]          # copy
    # 调用时把“上下文”插在最后一条 HumanMessage 前面
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1] + [ctx, history[-1]]
    else:
        history = [ctx] + history

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            resp = sa_agent.invoke({"messages": history})
            ai_msg = None
            msgs = resp.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    ai_msg = m
                    break
            if ai_msg is None:
                content = json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else str(resp)
                ai_msg = AIMessage(content=content)
            st.markdown(ai_msg.content)
    st.session_state.chat_messages.append(ai_msg)



# ========= RESULT RENDERERS & DIRECT RUN ===========

def _plot_survival_curves(surv_df):
    import numpy as np, matplotlib.pyplot as plt
    if not isinstance(surv_df, pd.DataFrame) or surv_df.empty:
        st.info("No survival trajectories available.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = list(surv_df.columns)[:min(5, len(surv_df.columns))]
    for c in cols:
        y = surv_df[c].values
        ax.step(range(len(y)), y, where="post", label=str(c))
    ax.set_title("Predicted Survival Trajectories")
    ax.set_xlabel("Time Bin"); ax.set_ylabel("Survival Probability"); ax.set_ylim(0.0, 1.05)
    ax.legend(title="Sample", fontsize=8)
    st.pyplot(fig)

def _compute_km(durations, events, limit_to_last_event=True, bin_width=0):
    import numpy as np
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    if bin_width and bin_width > 0:
        d = (np.floor(d / bin_width) * bin_width).astype(float)
    if limit_to_last_event and (e == 1).any():
        t_last = d[e == 1].max()
        keep = d <= t_last
        d, e = d[keep], e[keep]
    order = np.argsort(d); d, e = d[order], e[order]
    uniq = np.unique(d); n_at_risk = len(d); S = 1.0
    t_nodes, S_nodes = [0.0], [1.0]
    for tt in uniq:
        at = (d == tt); m = at.sum(); de = (e[at] == 1).sum()
        if de > 0 and n_at_risk > 0:
            S = S * (1.0 - de / n_at_risk)
            t_nodes.append(float(tt)); S_nodes.append(float(S))
        n_at_risk -= m
    return np.array(t_nodes), np.array(S_nodes)

def _plot_km_overall(df, limit_to_last_event=True, bin_width=0):
    import matplotlib.pyplot as plt
    t, S = _compute_km(df["duration"].values, df["event"].values,
                       limit_to_last_event=limit_to_last_event, bin_width=bin_width)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(t, S, where="post")
    ax.set_title("Kaplan–Meier (overall)")
    ax.set_xlabel("Time"); ax.set_ylabel("Survival Probability"); ax.set_ylim(0.0, 1.05)
    st.pyplot(fig)

def _coerce_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        if obj.get("type") == "dataframe" and isinstance(obj.get("data"), dict):
            data = obj["data"]
            if {"index", "columns", "data"}.issubset(data):
                df = pd.DataFrame(data["data"], columns=data["columns"])
                if len(data["index"]) == len(df):
                    df.index = data["index"]
                return df
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    if isinstance(obj, (list, tuple)):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None


def _render_results(res, df_for_km=None):
    """统一展示 key metrics / FI / 预测曲线 / KM。"""
    if not isinstance(res, dict):
        st.write(res)
        return
    if "error" in res:
        st.error(res["error"])
        if res.get("trace"):
            with st.expander("Traceback"):
                st.code(res["trace"])
        return
    metrics_block = res.get("metrics") if isinstance(res.get("metrics"), dict) else None
    metrics_source = metrics_block or {k: v for k, v in res.items() if isinstance(v, (int, float))}
    rows = [
        (k, float(v))
        for k, v in metrics_source.items()
        if isinstance(v, (int, float)) and pd.notna(v)
    ]
    if rows:
        st.subheader("Key metrics")

        metrics_df = (
            pd.DataFrame(rows, columns=["metric", "value"])
            .sort_values("metric", kind="stable")
            .reset_index(drop=True)
        )

        # 首选新版 dataframe API（紧凑 + 隐藏索引 + 数字格式），旧版则走 CSS 兜底
        try:
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
                height=min(320, 64 + 28 * len(metrics_df)),  # 控制整体高度，让行距更紧凑
                column_config={
                    "metric": st.column_config.TextColumn("Metric", width="medium"),
                    "value":  st.column_config.NumberColumn("Value", format="%.4f", width="small"),
                },
            )
        except Exception:
            # 兜底：用 table + CSS 压缩单元格内边距
            st.markdown(
                """
                <style>
                .metrics-table table { font-size: 0.92rem; }
                .metrics-table th, .metrics-table td {
                    padding-top: 4px !important;
                    padding-bottom: 4px !important;
                }
                .metrics-table thead th { background: rgba(255,255,255,0.04); }
                .metrics-table tbody tr:nth-child(odd) td { background: rgba(255,255,255,0.02); }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="metrics-table">', unsafe_allow_html=True)
            try:
                st.table(metrics_df.style.format({"value": "{:.4f}"}).hide(axis="index"))
            except Exception:
                metrics_df["value"] = metrics_df["value"].map(lambda x: f"{x:.4f}")
                st.table(metrics_df)
            st.markdown('</div>', unsafe_allow_html=True)

    notes = res.get("notes", [])
    if notes:
        st.info("\n\n".join(notes))

    artifacts = res.get("artifacts", {}) if isinstance(res.get("artifacts"), dict) else {}

    # --- survival curves ---
    def _first_dataframe(*candidates: Any) -> Optional[pd.DataFrame]:
        for cand in candidates:
            df = _coerce_dataframe(cand)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return None

    surv_df = _first_dataframe(
        artifacts.get("survival_curves"),
        res.get("survival_curves"),
        res.get("Surv_Test"),
    )
    if isinstance(surv_df, pd.DataFrame) and not surv_df.empty:
        st.subheader("Predicted Survival")
        _plot_survival_curves(surv_df)

    # --- FI ---
    fi_obj = (
        artifacts.get("feature_importance")
        or res.get("feature_importance")
        or res.get("texgi_importance")
        or res.get("fi_table")
    )
    fi_df = _coerce_dataframe(fi_obj)
    if fi_df is not None and not fi_df.empty:
        st.subheader("Feature Importance (TEXGISA)")
        st.dataframe(fi_df.head(10))
    # --- KM (基于原始 df 的 duration/event) ---
    if df_for_km is not None and {"duration", "event"}.issubset(df_for_km.columns):
        st.subheader("Kaplan–Meier")
        c1, c2 = st.columns(2)
        limit_flag = c1.checkbox("Limit x-axis to last event", value=True)
        bin_w = c2.number_input("KM bin width (0 = none)", 0, 999999, 0)
        _plot_km_overall(df_for_km, limit_to_last_event=limit_flag, bin_width=bin_w)

def _run_direct(
    algorithm_name,
    time_col,
    event_col,
    *,
    epochs=150,
    lr=0.01,
    batch_size=64,
    lambda_expert=None,
    preview=False,
    include_importance=True,
    show_status=True,
):
    """不经 LLM，直接调用你的工具函数 run_survival_analysis，然后渲染结果。"""
    cfg = dict(
        algorithm_name=algorithm_name,
        time_col=time_col,
        event_col=event_col,
        epochs=int(epochs),
        lr=float(lr),
        batch_size=int(batch_size),
    )
    if algorithm_name == "TEXGISA":
        # Restrict TEXGISA runs to the non-expert configuration.
        cfg["lambda_expert"] = 0.0
    spinner = st.spinner("Running...") if show_status else None
    if spinner:
        spinner.__enter__()
    try:
        res = run_survival_analysis(**cfg)
    finally:
        if spinner:
            spinner.__exit__(None, None, None)
    if show_status:
        st.success("Done.")
    # 从 DataManager 拿原始 df 传给 KM
    dm = st.session_state.data_manager
    get_df = getattr(dm, "get_current_dataframe", None)
    if callable(get_df):
        df_raw = get_df()
    else:
        df_raw = dm.get_data()
    if isinstance(res, dict) and not include_importance:
        res.pop("feature_importance", None)
        artifacts = res.get("artifacts") if isinstance(res.get("artifacts"), dict) else None
        if artifacts:
            for key in list(artifacts.keys()):
                if "importance" in key.lower():
                    artifacts.pop(key, None)
        details = res.get("details") if isinstance(res.get("details"), dict) else None
        if details:
            for key in list(details.keys()):
                if "importance" in key.lower():
                    details.pop(key, None)

    st.session_state["last_results"] = res
    st.session_state["last_results_df"] = df_raw
    return res

# ------------------------ main page ------------------------

def show():
    _style()
    _ensure_data_manager()

    st.title("🤖 Chat with Survival Analysis Agent")
    st.caption("You are here: Chat with Agent")
    st.write("Ask me to run an analysis, explain a model, or compare algorithms. "
             "You can upload data here and either let the AI call tools, or run models directly from the panel on the right.")

    # ---- session state ----
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_greeted" not in st.session_state:
        st.session_state.chat_greeted = False
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

    # ---- data uploader (RIGHT column) ----
    left, right = st.columns([0.68, 0.32], gap="large")

    with right:
        st.markdown("### 📁 Dataset")
        st.markdown('<div class="side-card">', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="chat_uploader")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.data_manager.load_data(df, uploaded.name)
                st.success(f"Loaded **{uploaded.name}** · shape = {df.shape[0]} × {df.shape[1]}")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        # read current summary
        dm_summary = st.session_state.data_manager.get_data_summary()
        if "error" in dm_summary:
            st.info("No dataset loaded yet.")
        else:
            n_rows = dm_summary.get("num_rows")
            n_cols = dm_summary.get("num_columns")
            st.write(f"**Rows**: {n_rows} · **Columns**: {n_cols}")
            with st.expander("Show columns"):
                cols = dm_summary.get("column_names", [])
                st.write(", ".join(map(str, cols)) if cols else "—")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- greeting depends on whether data is loaded ----
    has_data = "error" not in st.session_state.data_manager.get_data_summary()
    if not st.session_state.chat_greeted:
        if has_data:
            greet = (
                "Hello! Your dataset is loaded. Use the quick buttons to launch TEXGISA (no expert priors), CoxTime, DeepSurv, or DeepHit instantly. "
                "You can also type commands like 'run deepsurv time=duration event=event' for a guided workflow."
            )
        else:
            greet = (
                "Hello! Please upload a CSV on the right first. Once it is loaded you can run TEXGISA, CoxTime, DeepSurv, or DeepHit using the quick actions or chat commands."
            )
        st.session_state.chat_messages.append(AIMessage(content=greet))
        st.session_state.chat_greeted = True


    # ---- left: history & chat input ----
    with left:
        _render_history(st.session_state.chat_messages)

        pending = st.session_state.get("pending_action")
        if pending:
            st.warning(f"Pending confirmation: {pending['label']}. Reply 'yes' or 'no' to proceed.")

        # Quick chips (only meaningful after data is present)
        st.markdown("### ⚡ Quick Actions")
        with st.container():
            st.markdown('<div class="chip-row">', unsafe_allow_html=True)
            summary = st.session_state.data_manager.get_data_summary()
            tcol, ecol = (None, None)
            if "error" not in summary:
                tcol, ecol = _guess_cols_from_summary(summary)

            cols_qa = st.columns(4)
            with cols_qa[0]:
                if st.button("Run TEXGISA (no expert)", use_container_width=True, disabled=not has_data):
                    t = tcol or "duration"; e = ecol or "event"
                    label = f"Run TEXGISA (time={t}, event={e}, epochs=120)"
                    prompt = f"About to {label}. Reply **yes** to proceed or **no** to cancel."
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "TEXGISA", "time_col": t, "event_col": e, "epochs": 120, "include_importance": False},
                        prompt,
                        rerun=True,
                    )
            with cols_qa[1]:
                if st.button("Run CoxTime", use_container_width=True, disabled=not has_data):
                    t = tcol or "duration"; e = ecol or "event"
                    label = f"Run CoxTime (time={t}, event={e}, epochs=120)"
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "CoxTime", "time_col": t, "event_col": e, "epochs": 120},
                        f"About to {label}. Reply yes or no.",
                        rerun=True,
                    )
            with cols_qa[2]:
                if st.button("Run DeepSurv", use_container_width=True, disabled=not has_data):
                    t = tcol or "duration"; e = ecol or "event"
                    label = f"Run DeepSurv (time={t}, event={e}, epochs=120)"
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "DeepSurv", "time_col": t, "event_col": e, "epochs": 120},
                        f"About to {label}. Reply yes or no.",
                        rerun=True,
                    )
            with cols_qa[3]:
                if st.button("Run DeepHit", use_container_width=True, disabled=not has_data):
                    t = tcol or "duration"; e = ecol or "event"
                    label = f"Run DeepHit (time={t}, event={e}, epochs=120)"
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "DeepHit", "time_col": t, "event_col": e, "epochs": 120},
                        f"About to {label}. Reply yes or no.",
                        rerun=True,
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        # handle injected quick action
        if "__inject_user" in st.session_state:
            text = st.session_state.pop("__inject_user")
            _process_user_text(text)
            st.stop()

        # chat input — the agent will call tools if prompted properly
        user_text = st.chat_input("Type a message. E.g., run DeepHit time=duration event=event")
        if user_text:
            text = user_text.strip()
            low = text.lower()

            handled = False

            # Handle yes/no for any pending action first
            if st.session_state.get("pending_action"):
                if _run_pending_if_confirmed(text):
                    handled = True

            # 取列名猜测
            stt = _dataset_state()
            t_guess = "duration" if "duration" in stt["cols"] else (stt["cols"][0] if stt["cols"] else "duration")
            e_guess = "event" if "event" in stt["cols"] else (stt["cols"][1] if len(stt["cols"]) > 1 else "event")

            import re

            def _get_arg(pat, default=None, cast=str):
                m = re.search(pat, low)
                return cast(m.group(1)) if m else default

            # 专门支持“看我数据叫什么名”的小问题（不用 LLM，直接查）
            if not handled and re.search(r"(dataset).*name|name.*(dataset)", low):
                name = _dataset_state()["name"] or "No dataset loaded"
                with st.chat_message("assistant"):
                    st.markdown(f"Current dataset: **{name}**")
                st.session_state.chat_messages.append(AIMessage(content=f"Current dataset: {name}"))
                handled = True

            # -------- 轻量命令：直接跑你的代码（不经 LLM） --------
            if not handled and (
                low.startswith("preview")
                or "preview_fi" in low
                or "feature importance" in low
                or ("texgi" in low and "preview" in low)
            ):
                t = _get_arg(r"time(?:_col)?\s*=\s*([\w\.\-]+)", t_guess)
                e = _get_arg(r"event(?:_col)?\s*=\s*([\w\.\-]+)", e_guess)
                ep = _get_arg(r"epochs\s*=\s*(\d+)", 80, int)
                label = f"Preview TEXGISA feature importance (time={t}, event={e}, epochs={ep})"
                _queue_confirmation(label, _run_direct, {"algorithm_name": "TEXGISA", "time_col": t, "event_col": e, "epochs": ep, "preview": True, "include_importance": False})
                st.session_state.chat_messages.append(AIMessage(content=f"Queued: {label}. Reply yes or no to execute."))
                handled = True

            if not handled and (
                ("texgisa" in low or "texgi" in low)
                and (
                    low.startswith("run")
                    or low.startswith("train")
                    or " run " in low
                    or " train " in low
                )
            ):
                t = _get_arg(r"time(?:_col)?\s*=\s*([\w\.\-]+)", t_guess)
                e = _get_arg(r"event(?:_col)?\s*=\s*([\w\.\-]+)", e_guess)
                ep = _get_arg(r"epochs\s*=\s*(\d+)", 150, int)
                label = f"Run TEXGISA (time={t}, event={e}, epochs={ep})"
                _queue_confirmation(label, _run_direct, {"algorithm_name": "TEXGISA", "time_col": t, "event_col": e, "epochs": ep, "preview": False, "include_importance": False})
                st.session_state.chat_messages.append(AIMessage(content=f"Queued: {label}. Reply yes to start or no to cancel."))
                handled = True

            if handled:
                st.session_state.chat_messages.append(HumanMessage(content=text))
            else:
                # 其他自由文本 -> 走 LLM（但会被 B 步的上下文“约束”）
                _process_user_text(text)
                st.stop()



    # ---- right: Direct run (no LLM; guaranteed to execute) ----
    with right:
        st.markdown("### 🧪 Direct Run (no LLM)")
        st.caption("Execute TEXGISA, CoxTime, DeepSurv, or DeepHit instantly and view the structured results below.")
        st.markdown('<div class="side-card">', unsafe_allow_html=True)

        if has_data:
            cols = st.session_state.data_manager.get_data_summary().get("column_names", []) or []
            if not cols:
                st.warning("No feature columns detected in the dataset. Please verify the uploaded file contains duration/event columns.")
            with st.form("direct_run_form", clear_on_submit=False):
                algo = st.selectbox(
                    "Algorithm",
                    ["TEXGISA", "CoxTime", "DeepSurv", "DeepHit"],
                    help="Choose which survival algorithm to execute without the chat agent.",
                )
                time_options = cols or ["duration"]
                event_options = cols or ["event"]
                time_index = time_options.index("duration") if "duration" in time_options else 0
                event_index = event_options.index("event") if "event" in event_options else min(1, len(event_options) - 1)
                time_col = st.selectbox("Time column", options=time_options, index=time_index)
                event_col = st.selectbox("Event column", options=event_options, index=event_index)
                epochs = st.number_input("Epochs", 10, 1000, 150, step=10)
                lr = st.number_input("Learning rate", 1e-5, 1.0, 0.01, step=0.001, format="%.5f")
                batch_size = st.number_input("Batch size", 8, 512, 64, step=8)
                if algo == "TEXGISA":
                    st.caption("TEXGISA runs use the non-expert configuration (lambda_expert=0).")

                submitted = st.form_submit_button("Run now")
                if submitted:
                    algo_name = "TEXGISA" if algo.startswith("TEXGISA") else algo
                    label = f"Run {algo_name} (time={time_col}, event={event_col}, epochs={epochs}, lr={lr}, batch={batch_size})"
                    _queue_confirmation(
                        label,
                        _run_direct,
                        {
                            "algorithm_name": algo_name,
                            "time_col": time_col,
                            "event_col": event_col,
                            "epochs": epochs,
                            "lr": lr,
                            "batch_size": batch_size,
                            "preview": False,
                            "include_importance": False,
                        },
                    )
                    st.session_state.chat_messages.append(AIMessage(content=f"About to {label}. Reply yes or no to continue."))
        else:
            st.info("Upload a CSV to enable direct runs.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- full-width results panel ----
    st.markdown("### 📊 Results")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    last_res = st.session_state.get("last_results")
    df_for_km = st.session_state.get("last_results_df")
    if last_res:
        _render_results(last_res, df_for_km=df_for_km)
    else:
        st.info("Run any algorithm to see survival metrics, curves, and tables here.")
    st.markdown("</div>", unsafe_allow_html=True)
