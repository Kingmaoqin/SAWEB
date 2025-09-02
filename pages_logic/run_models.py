import streamlit as st
from sa_data_manager import DataManager

# Ensure DataManager exists in session state
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
manager = st.session_state.data_manager

st.title("Run Models")

# Sidebar options for preprocessing
st.sidebar.header("Preprocessing Options")
resample_rate = st.sidebar.number_input("Resample rate (Hz)", min_value=1, value=100)
image_resize = st.sidebar.number_input("Image resize (pixels)", min_value=1, value=224)

# Sensor file uploader
sensor_file = st.file_uploader(
    "Upload sensor data (CSV or JSON)",
    type=["csv", "json"],
)
if sensor_file is not None:
    sensor_names = manager.load_sensor(sensor_file)
    if sensor_names:
        st.write("Detected sensor file(s):")
        for name in sensor_names:
            st.write(name)

# Image file uploader
image_file = st.file_uploader(
    "Upload image data (ZIP)",
    type=["zip"],
)
if image_file is not None:
    image_names = manager.load_image(image_file)
    if image_names:
        st.write("Detected image file(s):")
        for name in image_names:
            st.write(name)
