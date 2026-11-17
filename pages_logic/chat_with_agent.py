# pages_logic/chat_with_agent.py — upload + real tool calls (agent & direct), clean UI

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# project agent & tools
from sa_tools import get_data_summary, run_survival_analysis, compare_algorithms

ALGORITHM_GUIDE: Dict[str, Dict[str, str]] = {
    "TEXGISA": {
        "summary": "Discrete-time survival analysis with time-aware feature explanations.",
        "best_for": "time-aware attribution reporting",
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
        "summary": "Neural model predicting survival across discrete time intervals.",
        "best_for": "non-linear time-to-event patterns",
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

    # User confirmed. Keep the page usable if a model rejects the selected data.
    st.session_state.pop("pending_action", None)
    try:
        with st.spinner("Running the selected analysis..."):
            result = pending["fn"](**pending["kwargs"])
        message = f"Completed: {pending['label']}"
        if isinstance(result, dict):
            st.session_state["last_results"] = result
    except Exception as exc:
        message = f"The analysis could not finish: {exc}"
        st.error(message)
    st.session_state.chat_messages.append(AIMessage(content=message))
    return True


def _style():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .side-card { 
            border: 1px solid #d1d5db;
            border-radius: 12px; 
            padding: 0.9rem 1.0rem; 
            background: #f8fafc;
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
            border: 1px solid #d1d5db;
            background: #f8fafc;
        }
        .chip-row .stButton>button:hover { background: #e5e7eb; }
        .stChatMessage { margin-bottom: 0.6rem; }
        .result-card {
            border: 1px solid #d1d5db;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            background: #ffffff;
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
    """Answer questions locally, with optional Groq for open-ended wording."""
    from utils.help_bot import answer

    st.session_state.chat_messages.append(HumanMessage(content=user_text))
    response = answer(user_text)
    st.session_state.chat_messages.append(AIMessage(content=response))
    with st.chat_message("assistant"):
        st.markdown(response)


def _reply(text: str):
    st.session_state.chat_messages.append(AIMessage(content=text))
    with st.chat_message("assistant"):
        st.markdown(text)


def _queue_model_from_text(text: str, summary: dict) -> bool:
    """Parse run/preview commands without depending on a hosted model."""
    lower = text.lower()
    preview = lower.startswith("preview") or "preview_fi" in lower
    model = next((name for name in ("CoxTime", "DeepSurv", "DeepHit", "TEXGISA")
                  if name.lower() in lower), None)
    if preview and model is None:
        model = "TEXGISA"
    if not model or not (preview or re.search(r"\b(run|train|start)\b", lower)):
        return False
    if preview and model != "TEXGISA":
        _reply("Feature-importance preview is available for TEXGISA. Select TEXGISA to run that preview.")
        return True
    if "error" in summary:
        _reply("Upload a CSV or load the synthetic demo before running a model.")
        return True
    columns = summary.get("column_names", [])
    guessed_time, guessed_event = _guess_cols_from_summary(summary)
    time_match = re.search(r"time(?:_col)?\s*=\s*([\w.\-]+)", lower)
    event_match = re.search(r"event(?:_col)?\s*=\s*([\w.\-]+)", lower)
    time_col = time_match.group(1) if time_match else guessed_time
    event_col = event_match.group(1) if event_match else guessed_event
    if not time_col or not event_col or time_col not in columns or event_col not in columns or time_col == event_col:
        _reply("Choose two different, existing columns for follow-up time and event status. "
               "For example: run CoxTime time=duration event=event epochs=20")
        return True
    epoch_match = re.search(r"epochs\s*=\s*(\d+)", lower)
    epochs = int(epoch_match.group(1)) if epoch_match else 20
    if not 1 <= epochs <= 1000:
        _reply("Choose an epoch count from 1 to 1000.")
        return True
    label = f"{'Preview' if preview else 'Run'} {model} (time={time_col}, event={event_col}, epochs={epochs})"
    _queue_confirmation(label, _run_direct, {
        "algorithm_name": model,
        "time_col": time_col,
        "event_col": event_col,
        "epochs": epochs,
        "preview": preview,
    })
    _reply(f"Ready to {label}. Reply **yes** to start or **no** to cancel.")
    return True


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

        st.download_button("Download metrics CSV", metrics_df.to_csv(index=False).encode("utf-8"),
                           file_name="survival_metrics.csv", mime="text/csv")

        # Render the compact metrics table, with a plain table as fallback.
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
                .metrics-table thead th { background: #f8fafc; }
                .metrics-table tbody tr:nth-child(odd) td { background: #ffffff; }
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
        st.download_button("Download survival curves CSV", surv_df.to_csv().encode("utf-8"),
                           file_name="predicted_survival.csv", mime="text/csv")

    # --- FI ---
    fi_obj = next((candidate for candidate in (
        artifacts.get("feature_importance"), res.get("feature_importance"),
        res.get("texgi_importance"), res.get("fi_table")) if candidate is not None), None)
    fi_df = _coerce_dataframe(fi_obj)
    if fi_df is not None and not fi_df.empty:
        st.subheader("Feature Importance (TEXGISA)")
        st.dataframe(fi_df.head(10))
        st.download_button("Download feature importance CSV", fi_df.to_csv(index=False).encode("utf-8"),
                           file_name="texgisa_feature_importance.csv", mime="text/csv")
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
    if preview and algorithm_name == "TEXGISA":
        cfg["epochs"] = min(cfg["epochs"], 20)
    spinner = st.spinner("Running...") if show_status else None
    if spinner:
        spinner.__enter__()
    try:
        res = run_survival_analysis(**cfg)
    finally:
        if spinner:
            spinner.__exit__(None, None, None)
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(res["error"])
    if show_status:
        st.success("Done.")
    # 从 DataManager 拿原始 df 传给 KM
    dm = st.session_state.data_manager
    get_df = getattr(dm, "get_current_dataframe", None)
    if callable(get_df):
        df_raw = get_df()
    else:
        df_raw = dm.get_data()
    if isinstance(res, dict) and not include_importance and algorithm_name != "TEXGISA":
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

    from utils.user_center import record_result
    record_result(algorithm_name, res, "Chat Direct Run")
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

        demo_path = Path(__file__).resolve().parents[1] / "demo_data" / "synthetic_survival_demo.csv"
        if st.button("Load synthetic demo in chat", use_container_width=True):
            demo_data = pd.read_csv(demo_path)
            st.session_state.data_manager.load_data(demo_data, demo_path.name)
            st.session_state.pop("last_results", None)
            st.session_state.pop("last_results_df", None)
            st.success(f"Loaded {len(demo_data)} synthetic records.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="chat_uploader")
        if uploaded is not None:
            try:
                raw = uploaded.getvalue()
                fingerprint = hashlib.sha256(raw).hexdigest()
                if st.session_state.get("chat_upload_fingerprint") != fingerprint:
                    df = pd.read_csv(io.BytesIO(raw))
                    st.session_state.data_manager.load_data(df, uploaded.name)
                    st.session_state["chat_upload_fingerprint"] = fingerprint
                    st.session_state.pop("last_results", None)
                    st.session_state.pop("last_results_df", None)
                    st.session_state.pop("qa_time_col", None)
                    st.session_state.pop("qa_event_col", None)
                    st.session_state.pop("qa_time_select", None)
                    st.session_state.pop("qa_event_select", None)
                st.success(f"Loaded **{uploaded.name}** · shape = {st.session_state.data_manager.get_data().shape[0]} × {st.session_state.data_manager.get_data().shape[1]}")
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
            cols = []
            if "error" not in summary:
                tcol, ecol = _guess_cols_from_summary(summary)
                cols = summary.get("column_names", []) or []

            # Allow the user to pick the time/event columns used by the quick buttons.
            st.caption("Select time/event columns for quick runs and confirmations.")
            qc1, qc2 = st.columns(2)
            default_time = tcol or ("duration" if "duration" in cols else (cols[0] if cols else "duration"))
            default_event = ecol or ("event" if "event" in cols else (cols[1] if len(cols) > 1 else default_time))

            if st.session_state.get("qa_time_col") not in cols:
                st.session_state.qa_time_col = default_time
                st.session_state["qa_time_select"] = default_time
            if (st.session_state.get("qa_event_col") not in cols
                    or st.session_state.get("qa_event_col") == st.session_state.qa_time_col):
                st.session_state.qa_event_col = default_event
                st.session_state["qa_event_select"] = default_event

            with qc1:
                st.session_state.qa_time_col = st.selectbox(
                    "Time column",
                    options=cols or [default_time],
                    index=(cols.index(st.session_state.qa_time_col)
                           if cols and st.session_state.qa_time_col in cols else 0),
                    key="qa_time_select",
                )
            with qc2:
                st.session_state.qa_event_col = st.selectbox(
                    "Event column",
                    options=cols or [default_event],
                    index=(cols.index(st.session_state.qa_event_col)
                           if cols and st.session_state.qa_event_col in cols else 0),
                    key="qa_event_select",
                )

            btn_disabled = (not has_data or not st.session_state.qa_time_col or not st.session_state.qa_event_col
                            or st.session_state.qa_time_col == st.session_state.qa_event_col)

            cols_qa = st.columns(4)
            with cols_qa[0]:
                if st.button("Run TEXGISA (no expert)", use_container_width=True, disabled=btn_disabled):
                    t = st.session_state.qa_time_col; e = st.session_state.qa_event_col
                    label = f"Run TEXGISA (time={t}, event={e}, epochs=20)"
                    prompt = f"About to {label}. Reply **yes** to proceed or **no** to cancel."
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "TEXGISA", "time_col": t, "event_col": e, "epochs": 20, "include_importance": True},
                        prompt,
                        rerun=True,
                    )
            with cols_qa[1]:
                if st.button("Run CoxTime", use_container_width=True, disabled=btn_disabled):
                    t = st.session_state.qa_time_col; e = st.session_state.qa_event_col
                    label = f"Run CoxTime (time={t}, event={e}, epochs=20)"
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "CoxTime", "time_col": t, "event_col": e, "epochs": 20},
                        f"About to {label}. Reply yes or no.",
                        rerun=True,
                    )
            with cols_qa[2]:
                if st.button("Run DeepSurv", use_container_width=True, disabled=btn_disabled):
                    t = st.session_state.qa_time_col; e = st.session_state.qa_event_col
                    label = f"Run DeepSurv (time={t}, event={e}, epochs=20)"
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "DeepSurv", "time_col": t, "event_col": e, "epochs": 20},
                        f"About to {label}. Reply yes or no.",
                        rerun=True,
                    )
            with cols_qa[3]:
                if st.button("Run DeepHit", use_container_width=True, disabled=btn_disabled):
                    t = st.session_state.qa_time_col; e = st.session_state.qa_event_col
                    label = f"Run DeepHit (time={t}, event={e}, epochs=20)"
                    _queue_and_prompt(
                        label,
                        _run_direct,
                        {"algorithm_name": "DeepHit", "time_col": t, "event_col": e, "epochs": 20},
                        f"About to {label}. Reply yes or no.",
                        rerun=True,
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        user_text = st.chat_input("Ask a question or type: run DeepHit time=duration event=event epochs=20")
        if user_text and user_text.strip():
            text = user_text.strip()
            lower = text.lower()
            st.session_state.chat_messages.append(HumanMessage(content=text))
            if st.session_state.get("pending_action"):
                _run_pending_if_confirmed(text)
            elif lower in {"yes", "y", "no", "n"}:
                _reply("No analysis is waiting for confirmation. Choose a model or enter a run command.")
            elif re.search(r"(dataset).*name|name.*(dataset)", lower):
                name = _dataset_state()["name"] or "No dataset loaded"
                _reply(f"Current dataset: **{name}**")
            elif "compare" in lower and ("model" in lower or "algorithm" in lower):
                comparison = compare_algorithms()
                lines = ["| Model | Main use |", "|---|---|"]
                for model, use in comparison["recommendations"].items():
                    lines.append(f"| {use} | {model} |")
                _reply("\n".join(lines))
            elif "column" in lower and ("data" in lower or "dataset" in lower):
                summary = st.session_state.data_manager.get_data_summary()
                _reply("Columns: " + ", ".join(map(str, summary.get("column_names", [])))
                       if "error" not in summary else "Upload a CSV first to inspect its columns.")
            elif _queue_model_from_text(text, st.session_state.data_manager.get_data_summary()):
                pass
            else:
                # The question helper answers common concepts locally and uses Groq
                # for other wording when a key is configured.
                from utils.help_bot import answer
                _reply(answer(text))


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
                epochs = st.number_input("Epochs", 2, 1000, 20, step=2)
                lr = st.number_input("Learning rate", 1e-5, 1.0, 0.01, step=0.001, format="%.5f")
                batch_size = st.number_input("Batch size", 8, 512, 64, step=8)
                if algo == "TEXGISA":
                    st.caption("TEXGISA runs use the non-expert configuration (lambda_expert=0).")

                invalid_mapping = time_col == event_col
                if invalid_mapping:
                    st.warning("Choose different time and event columns.")
                submitted = st.form_submit_button("Run now", disabled=invalid_mapping)
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
                            "include_importance": True,
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
