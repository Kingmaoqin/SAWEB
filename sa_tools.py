# sa_tools.py (New full content)

import json
from typing import Any, Dict

import streamlit as st
import pandas as pd
from models import coxtime, deepsurv, deephit

# Prefer the modern MySA implementation; fall back to the legacy TEXGISA entry point when present.
try:
    from models.mysa import run_mysa as run_texgisa  # Keep legacy alias for compatibility.
except Exception:
    try:
        from models.texgisa import run_texgisa  # Legacy implementation if still available.
    except Exception:
        run_texgisa = None

def suggest_next_actions() -> dict:
    """
    Checks the current application state (e.g., if data is loaded) and suggests relevant next actions.
    This should be the first tool called in most conversations.
    """
    # Ensure the shared DataManager exists in the current session.
    if "data_manager" not in st.session_state:
        from sa_data_manager import DataManager
        st.session_state.data_manager = DataManager()

    data = st.session_state.data_manager.get_data()
    if data is None:
        return {
            "message": "I see that no data has been uploaded yet. Please go to the 'Run Models' page to upload your clinical dataset.",
            "actions": []
        }

    # Reuse the session-level DataManager to avoid reloading state between tool calls.
    summary = st.session_state.data_manager.get_data_summary()
    file_name = summary.get("file_name", "your dataset")

    actions = [
        {
            "label": "Summarize Data",
            "tool_name": "get_data_summary",
            "tool_args": {}
        },
        {
            "label": "Run CoxTime Model",
            "tool_name": "run_survival_analysis",
            "tool_args": {"algorithm_name": "CoxTime"}
        },
        {
            "label": "Run DeepSurv Model",
            "tool_name": "run_survival_analysis",
            "tool_args": {"algorithm_name": "DeepSurv"}
        },
        {
            "label": "Run DeepHit Model",
            "tool_name": "run_survival_analysis",
            "tool_args": {"algorithm_name": "DeepHit"}
        }
    ]
    if run_texgisa is not None:
        actions.append({
            "label": "Run TEXGISA (MySA)",
            "tool_name": "run_survival_analysis",
            "tool_args": {"algorithm_name": "TEXGISA"}
        })

    return {
        "message": f"Great! I see you've loaded the file '{file_name}'. It has {summary.get('num_rows')} rows and {summary.get('num_columns')} columns. What would you like to do with it?",
        "actions": actions
    }

def _serialise_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a JSON-friendly payload that can be reconstructed as a DataFrame."""

    data = df.to_dict(orient="split")
    return {
        "type": "dataframe",
        "data": {
            "index": list(map(_ensure_json_primitive, data["index"])),
            "columns": [str(c) for c in data["columns"]],
            "data": [[_ensure_json_primitive(v) for v in row] for row in data["data"]],
        },
    }


def _ensure_json_primitive(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas scalars into plain Python types."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _ensure_json_structure(obj: Any) -> Any:
    """Recursively convert nested structures into JSON-friendly representations."""

    if isinstance(obj, dict):
        return {str(k): _ensure_json_structure(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_json_structure(v) for v in obj]
    return _ensure_json_primitive(obj)


def run_survival_analysis(
    algorithm_name: str,
    time_col: str,
    event_col: str,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    **kwargs
) -> dict:
    """
    Runs a specified survival analysis model on the currently loaded dataset.
    """
    # Ensure the shared DataManager exists in the current session; fall back to the
    # most recently used manager when tools run in a background thread without a
    # ScriptRunContext.
    try:
        dm = st.session_state.get("data_manager")
    except Exception:
        dm = None

    if dm is None:
        from sa_data_manager import get_shared_manager

        # In background threads, allow reuse of the last loaded dataset; otherwise
        # a fresh, empty manager is returned and we surface a clear error below.
        dm = get_shared_manager(allow_global_fallback=True)
    else:
        # keep the global mirror warm for non-Streamlit threads
        from sa_data_manager import _remember_manager  # type: ignore

        _remember_manager(dm)

    data = dm.get_data()
    if data is None:
        return {"error": "No data found. Please ask the user to upload a dataset on the 'Run Models' page first."}

    if time_col not in data.columns or event_col not in data.columns:
        return {"error": f"Invalid column names. Ensure '{time_col}' and '{event_col}' exist in the data. Available columns are: {data.columns.tolist()}"}

    # Standardise column names so downstream training code receives the canonical schema.
    df = data.rename(columns={time_col: "duration", event_col: "event"}).copy()
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]

    # Populate the base configuration shared by all algorithms.
    config = {
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "time_col": "duration",
        "event_col": "event",
        "feature_cols": feature_cols,
    }

    # Forward optional TEXGISA parameters when provided by the caller.
    passthrough = [
        "lambda_smooth", "lambda_expert", "expert_rules",
        "ig_steps", "latent_dim", "extreme_dim",
        "gen_epochs", "gen_batch", "gen_lr", "gen_alpha_dist",
        "num_intervals", "n_bins"
    ]
    for k in passthrough:
        if k in kwargs and kwargs[k] is not None:
            config[k] = kwargs[k]

    try:
        algo = algorithm_name.lower()
        if algo == "coxtime":
            results = coxtime.run_coxtime(df, config)
        elif algo == "deepsurv":
            results = deepsurv.run_deepsurv(df, config)
        elif algo == "deephit":
            config.setdefault("num_intervals", 50)
            results = deephit.run_deephit(df, config)
        elif algo in ("texgisa", "mysa", "texgisa (mysa)"):
            if run_texgisa is None:
                return {"error": "TEXGISA/MySA not available. Please place models/mysa.py."}
            results = run_texgisa(df, config)
        else:
            results = {"error": f"Unknown algorithm name: {algorithm_name}"}

        if isinstance(results, dict) and "error" in results:
            return results

        payload: Dict[str, Any] = {
            "algorithm": algorithm_name,
            "time_column": time_col,
            "event_column": event_col,
            "feature_columns": [str(c) for c in feature_cols],
            "metrics": {},
        }

        artifacts: Dict[str, Any] = {}

        for key, value in results.items():
            if isinstance(value, (int, float)):
                payload["metrics"][key] = _ensure_json_primitive(value)
            elif isinstance(value, str):
                payload.setdefault("notes", []).append(f"{key}: {value}")
            elif isinstance(value, pd.DataFrame):
                mapped_key = key
                if key.lower() in {"surv_test", "survival_curves", "surv"}:
                    mapped_key = "survival_curves"
                elif "texgi" in key.lower() or "importance" in key.lower():
                    mapped_key = "feature_importance"
                artifacts[mapped_key] = _serialise_dataframe(value)
            elif isinstance(value, dict):
                payload.setdefault("details", {})[key] = _ensure_json_structure(value)
            elif isinstance(value, (list, tuple)):
                payload.setdefault("details", {})[key] = _ensure_json_structure(value)

        if artifacts:
            payload["artifacts"] = artifacts

        return payload

    except Exception as e:
        import traceback
        return {"error": f"An error occurred during analysis: {str(e)}", "trace": traceback.format_exc()}


def get_algorithm_explanation(algorithm_name: str) -> dict:
    """Provides a detailed explanation of a survival analysis algorithm."""
    explanations = {
        "coxtime": {
            "name": "CoxTime",
            "summary": "A time-dependent Cox model that uses neural networks to learn time-varying coefficient effects.",
            "use_case": "Best for situations where the effect of a covariate (e.g., a drug) changes over time."
        },
        "deepsurv": {
            "name": "DeepSurv",
            "summary": "A deep learning extension of the standard Cox Proportional Hazards model.",
            "use_case": "Ideal for capturing complex, non-linear relationships between patient features and survival risk."
        },
        "deephit": {
            "name": "DeepHit",
            "summary": "A deep learning model designed for survival analysis with competing risks.",
            "use_case": "Use when there are multiple possible event types (e.g., cardiac death vs. stroke) and you want to model the probability of each."
        },
        "texgisa": {
            "name": "TEXGISA (MySA)",
            "summary": "Generative survival analysis that couples multimodal encoders with TEXGISA explanations and optional expert priors.",
            "use_case": "Choose when you need end-to-end multimodal training or when domain experts provide priors that should regularise the hazard estimates."
        },
        "mysa": {
            "name": "TEXGISA (MySA)",
            "summary": "Generative survival analysis that couples multimodal encoders with TEXGISA explanations and optional expert priors.",
            "use_case": "Choose when you need end-to-end multimodal training or when domain experts provide priors that should regularise the hazard estimates."
        },
        "survival analysis": {
            "name": "Survival Analysis",
            "summary": "A family of statistical methods for modelling time-to-event outcomes (e.g., time until relapse or equipment failure).",
            "use_case": "Use to estimate risk over time, compare treatment groups, or handle censored observations where the event has not yet occurred."
        }
    }
    key = algorithm_name.lower()
    if key == "texgisa (mysa)":
        key = "texgisa"
    # Provide a helpful fallback instead of an error so generic questions still receive an answer.
    if key not in explanations:
        return {
            "name": algorithm_name,
            "summary": "Survival analysis studies the time until an event occurs (like death or failure) while handling censored observations.",
            "use_case": "Use it when you need to estimate risk or survival probability over time, compare cohorts, or model how features influence event timing."
        }
    return explanations[key]

def compare_algorithms() -> dict:
    """Provides a comparison of the available survival analysis algorithms."""
    comparison_data = {
        "headers": ["Metric", "CoxTime", "DeepSurv", "DeepHit", "TEXGISA (MySA)"],
        "rows": [
            ["Time Complexity", "O(n^2)", "O(n)", "O(nm)", "High (multimodal training)"],
            ["Handles Competing Risks", "No", "No", "Yes", "No"],
            ["Handles Time-Varying Effects", "Yes", "No", "No", "Yes"],
            ["Multimodal Support", "Tabular only", "Tabular only", "Tabular only", "Tabular + raw images/sensors"],
            ["Expert Priors", "No", "No", "No", "Yes"],
            ["Interpretability", "Medium", "Low", "Low", "High via TEXGISA"]
        ],
        "recommendations": {
            "Dynamic Effects": "CoxTime",
            "Nonlinear Patterns": "DeepSurv",
            "Multiple Outcomes": "DeepHit",
            "Multimodal + Explanations": "TEXGISA (MySA)"
        }
    }
    return comparison_data

def explain_hyperparameter(param_name: str) -> dict:
    """Explains a model hyperparameter."""
    param_descriptions = {
        "learning_rate": {
            "name": "Learning Rate (lr)",
            "description": "Controls how much to change the model in response to the estimated error each time the model weights are updated.",
            "impact": "A low learning rate may result in slow training, while a high one may cause the model to converge too quickly to a suboptimal solution."
        },
        "batch_size": {
            "name": "Batch Size",
            "description": "The number of training examples utilized in one iteration (or epoch).",
            "impact": "A larger batch size can lead to faster training but requires more memory. A smaller batch size can offer a regularizing effect but makes training less stable."
        },
        "epochs": {
            "name": "Epochs",
            "description": "The number of times the entire training dataset is passed forward and backward through the neural network.",
            "impact": "Too few epochs can lead to underfitting, while too many can lead to overfitting."
        }
    }
    return param_descriptions.get(param_name.lower(), {"error": f"Parameter '{param_name}' not found."})

def get_data_summary() -> dict:
    """Retrieves a summary of the currently loaded dataset."""
    try:
        dm = st.session_state.get("data_manager")
    except Exception:
        dm = None

    if dm is None:
        from sa_data_manager import get_shared_manager

        dm = get_shared_manager(allow_global_fallback=True)
    else:
        from sa_data_manager import _remember_manager  # type: ignore

        _remember_manager(dm)

    if dm.get_data() is None:
        return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}
    return dm.get_data_summary()
