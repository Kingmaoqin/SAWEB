# sa_tools.py (New full content)

import json
import streamlit as st
import pandas as pd
from models import coxtime, deepsurv, deephit

# Prefer the new MySA; fall back to legacy texgisa; allow missing
try:
    from models.mysa import run_mysa as run_texgisa  # MySA 兼容旧入口名
except Exception:
    try:
        from models.texgisa import run_texgisa       # 老实现（如果你仍然保留）
    except Exception:
        run_texgisa = None

def suggest_next_actions() -> dict:
    """
    Checks the current application state (e.g., if data is loaded) and suggests relevant next actions.
    This should be the first tool called in most conversations.
    """
    # 确保 DataManager 存在
    if "data_manager" not in st.session_state:
        from sa_data_manager import DataManager
        st.session_state.data_manager = DataManager()

    data = st.session_state.data_manager.get_data()
    if data is None:
        return {
            "message": "I see that no data has been uploaded yet. Please go to the 'Run Models' page to upload your clinical dataset.",
            "actions": []
        }

    # 用会话里的 DataManager 拿摘要（不要再调模块级 sa_data_manager）
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
    # 确保 DataManager 存在
    if "data_manager" not in st.session_state:
        from sa_data_manager import DataManager
        st.session_state.data_manager = DataManager()

    data = st.session_state.data_manager.get_data()
    if data is None:
        return {"error": "No data found. Please ask the user to upload a dataset on the 'Run Models' page first."}

    if time_col not in data.columns or event_col not in data.columns:
        return {"error": f"Invalid column names. Ensure '{time_col}' and '{event_col}' exist in the data. Available columns are: {data.columns.tolist()}"}

    # 统一列名
    df = data.rename(columns={time_col: "duration", event_col: "event"}).copy()
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]

    # 通用配置
    config = {
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "time_col": "duration",
        "event_col": "event",
        "feature_cols": feature_cols,
    }

    # 透传 MySA/TEXGISA 的可选参数（如果调用方传了就生效）
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

        # 返回给 Agent 的场景只保留易序列化的内容（指标等）
        metrics = {k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))}
        return metrics

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
        }
    }
    return explanations.get(algorithm_name.lower(), {"error": "Algorithm not found."})

def compare_algorithms() -> dict:
    """Provides a comparison of the available survival analysis algorithms."""
    comparison_data = {
        "headers": ["Metric", "CoxTime", "DeepSurv", "DeepHit"],
        "rows": [
            ["Time Complexity", "O(n²)", "O(n)", "O(nm)"],
            ["Handles Competing Risks", "No", "No", "Yes"],
            ["Handles Time-Varying Effects", "Yes", "No", "No"],
            ["Interpretability", "Medium", "Low", "Low"]
        ],
        "recommendations": {
            "Dynamic Effects": "CoxTime",
            "Nonlinear Patterns": "DeepSurv",
            "Multiple Outcomes": "DeepHit"
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
    dm = st.session_state.get("data_manager")
    if dm is None or dm.get_data() is None:
        return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}
    return dm.get_data_summary()
