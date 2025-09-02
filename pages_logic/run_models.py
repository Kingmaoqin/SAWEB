import pandas as pd
import streamlit as st

from sa_data_manager import DataManager


def _read_dataframe(uploaded_file) -> pd.DataFrame:
    """Read uploaded file into a DataFrame."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if uploaded_file.name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Unsupported file type")


def show() -> None:
    """Render the model-running page with data summaries."""
    st.header("Model Training")

    uploaded = st.file_uploader("Upload dataset", type=["csv", "parquet"])
    if uploaded is None:
        return

    df = _read_dataframe(uploaded)

    data_manager: DataManager = st.session_state.data_manager
    data_manager.load_data(df, uploaded.name)

    # Consolidated summary
    st.subheader("Dataset Summary")
    st.json(data_manager.get_data_summary())

    # Sensor stats
    st.subheader("Sensor Summary")
    sensor_summary = data_manager.get_sensor_summary()
    if isinstance(sensor_summary, dict) and sensor_summary.get("error"):
        st.info(sensor_summary["error"])
    else:
        st.table(pd.DataFrame(sensor_summary))  # type: ignore[arg-type]

    # Image stats
    st.subheader("Image Summary")
    image_summary = data_manager.get_image_summary()
    if isinstance(image_summary, dict) and image_summary.get("error"):
        st.info(image_summary["error"])
    else:
        for item in image_summary:  # type: ignore[assignment]
            st.write(f"Column: {item['column']}")
            st.write(f"Shape: {item['shape']}")
            st.image(item["image"], width=100)
