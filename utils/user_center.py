"""Session-scoped dataset and analysis summaries for the sidebar."""

from datetime import datetime
from numbers import Real

import pandas as pd
import streamlit as st


MODELS = ("CoxTime", "DeepSurv", "DeepHit", "TEXGISA")
SECTIONS = ("Our Datasets", "Our Experiments", "Our Results", "History", "Favorite Models")


def _metrics(result):
    source = result.get("metrics", result) if isinstance(result, dict) else {}
    return {key: float(value) for key, value in source.items()
            if isinstance(value, Real) and pd.notna(value) and not isinstance(value, bool)}


def record_result(model, result, source):
    """Save a small run summary in this browser session, without patient data."""
    history = st.session_state.setdefault("analysis_history", [])
    history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model,
        "source": source,
        "metrics": _metrics(result),
    })
    st.session_state["analysis_history"] = history[-20:]


def show(section=None):
    """Render real summaries from the current browser session."""
    if section is None:
        section = st.selectbox("Section", SECTIONS)
    st.subheader(section)

    if section == "Our Datasets":
        manager = st.session_state.get("data_manager")
        data = manager.get_data() if manager is not None else None
        if data is None:
            data = st.session_state.get("clinical_data")
        if not isinstance(data, pd.DataFrame):
            st.info("No dataset loaded in this session.")
            return
        st.write(f"{len(data)} rows · {len(data.columns)} columns")
        st.caption(", ".join(map(str, data.columns)))
        st.dataframe(data.head(5), hide_index=True, use_container_width=True)
        st.download_button("Download current CSV", data.to_csv(index=False).encode("utf-8"),
                           file_name="current_survival_data.csv", mime="text/csv")
        return

    history = st.session_state.get("analysis_history", [])
    if section == "Favorite Models":
        selected = st.multiselect("Models to keep handy", MODELS,
                                  default=st.session_state.get("favorite_models", []))
        st.session_state["favorite_models"] = selected
        if selected:
            st.caption("Saved for this browser session: " + ", ".join(selected))
        return

    if not history:
        st.info("Run a model to see results from this browser session.")
        return

    if section == "Our Results":
        latest = history[-1]
        st.caption(f"Latest: {latest['model']} · {latest['time']}")
        if latest["metrics"]:
            st.dataframe(pd.DataFrame(latest["metrics"].items(), columns=["Metric", "Value"]),
                         hide_index=True, use_container_width=True)
        else:
            st.info("This run did not return numeric metrics.")
        return

    runs = list(reversed(history))
    if section == "Our Experiments":
        runs = runs[:5]
    if section in ("Our Experiments", "History"):
        st.dataframe(pd.DataFrame([{"Time": item["time"], "Model": item["model"],
                                   "Page": item["source"]} for item in runs]),
                     hide_index=True, use_container_width=True)
