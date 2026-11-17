"""Plain English, reliable on-screen guidance for the survival analysis app."""

import os
import re

import requests
import streamlit as st


SYSTEM_PROMPT = (
    "You are the help assistant for a survival analysis web app. Answer in clear English "
    "in two to four short sentences. Explain terms on first use. The app accepts a CSV "
    "with duration (time observed), event (1=observed event, 0=censored), and feature columns. "
    "It offers CoxTime, DeepSurv, DeepHit, and TEXGISA. Do not claim to have run a model "
    "or read a dataset. Do not give a clinical diagnosis or invent metrics."
)


def _local_answer(question: str) -> str | None:
    """Answer the core demo questions without relying on a network model."""
    q = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    if "survival analysis" in q or "time to event" in q:
        return ("Survival analysis studies how long it takes for an event to occur, such as a relapse "
                "or device failure. It uses each person's observed follow-up time and whether the "
                "event happened. It can include people whose event was not observed before follow-up ended.")
    if "censor" in q:
        return ("Censoring means the event was not observed during the available follow-up. "
                "The person's time still contributes information up to their last observation; "
                "in the CSV, event=0 marks a censored record.")
    if "c index" in q or "concordance" in q:
        return ("The C-index measures how often a model ranks two comparable people in the right "
                "order of event time. A value near 0.5 is roughly random ranking; a higher value "
                "means better ranking on the evaluated data.")
    if "brier" in q:
        return ("The Brier score compares predicted event probabilities with observed outcomes "
                "at a time point. Lower values are better; the integrated Brier score averages "
                "this error over the evaluated time range.")
    if "kaplan" in q or "meier" in q:
        return ("A Kaplan–Meier curve shows the fraction of the observed group that remains "
                "event-free over time, while accounting for censoring. Compare its shape with "
                "predicted survival curves to inspect the model's behavior.")
    if "event" in q and ("column" in q or "mean" in q or "0" in q or "1" in q):
        return ("The event column records whether the event occurred during follow-up: "
                "1 means observed event, and 0 means censored. Choose that column under "
                "Column Mapping before training.")
    if "duration" in q or ("time" in q and "column" in q):
        return ("The duration column is the observed follow-up time for each row. "
                "Use one consistent unit, such as months, throughout the CSV.")
    if "coxtime" in q:
        return ("CoxTime is a survival model that allows a feature's effect on event risk to "
                "change over time. Select it under Model Training to inspect its ranking metrics "
                "and predicted survival curves.")
    if "deepsurv" in q:
        return ("DeepSurv uses a neural network to estimate a person's relative event risk "
                "from their features. Its C-index describes how well it ranks comparable people.")
    if "deephit" in q:
        return ("DeepHit predicts event probabilities across discrete time intervals. "
                "Its output can be displayed as a predicted survival curve for each person.")
    if "texgisa" in q or "feature importance" in q or "expert prior" in q:
        return ("TEXGISA predicts survival across time intervals and provides time-aware feature "
                "importance. In Model Training, you can preview feature importance, add expert "
                "rules, and inspect the resulting curves.")
    if "upload" in q or "csv" in q or "dataset" in q:
        return ("Open Model Training and upload a CSV, or click Load synthetic survival demo. "
                "Map the duration and event columns, keep the feature columns you want, then "
                "choose a model and click its training button.")
    if "train" in q or "run model" in q or "algorithm" in q:
        return ("In Model Training, load data, select the duration and event columns, and choose "
                "CoxTime, DeepSurv, DeepHit, or TEXGISA. Start a short run with a few epochs, "
                "then review the metrics and survival charts below.")
    return None


def _groq_key() -> str | None:
    try:
        if st.secrets.load_if_toml_exists() and st.secrets.get("GROQ_API_KEY"):
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip() or None


def answer(question: str) -> str:
    """Return a useful answer even when the optional language model is unavailable."""
    question = question.strip()
    if not question:
        return "Ask a question about survival analysis or how to use this page."
    local = _local_answer(question)
    if local:
        return local

    key = _groq_key()
    if key:
        try:
            response = requests.post(
                os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
                + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    "temperature": 0.2,
                    "max_tokens": 180,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                },
                timeout=12,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content:
                return content
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
            pass

    return ("I can explain survival time, censoring, the C-index, the Brier score, "
            "the four models, and how to upload and train a CSV. Ask about one of those topics.")
