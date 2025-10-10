# run_models.py (FINAL VERSION - Supports both Manual and AI modes)

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from models import coxtime, deepsurv, deephit
from models.texgisa import run_texgisa

def show():
    st.title("Survival Analysis Workbench")
    
    # Section 1: Step-by-Step Guidance
    with st.expander("📘 Step-by-Step Guide", expanded=True):
        st.markdown("""
        **Data Analysis Workflow:**
        1.  **Upload Data**: Use the section below to upload your CSV file.
        2.  **Choose Your Path**:
            * **AI Assistant**: Click 'Go to Chat with Agent' to use the conversational AI.
            * **Manual Analysis**: Configure parameters in 'Configure Analysis Parameters' and click 'Start Survival Analysis'.
        """)
    
    # Section 2: Data Upload
    with st.expander("📁 Upload Clinical Data", expanded=True):
        uploaded_file = st.file_uploader("Select CSV file", type=["csv"],
                                         help="Dataset must contain time-to-event and event indicator columns")
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                # This key is used to show/hide the manual analysis section
                st.session_state["clinical_data"] = data
                
                # This correctly saves data for the AI Agent's use across pages
                st.session_state.data_manager.load_data(data, uploaded_file.name)
                
                st.success(f"✅ Data from '{uploaded_file.name}' successfully loaded!")
                
                # --- RETAINED FEATURE: Navigate to the AI chat page ---
                st.info("Your data is ready. You can now chat with the AI assistant about it.")
                if st.button("💬 Go to Chat with Agent", type="primary"):
                    st.session_state["current_page"] = "Chat with Agent"
                    st.session_state.chat_messages = []
                    st.rerun()
                
                st.subheader("Clinical Data Preview")
                st.dataframe(data.head(3), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Data loading failed: {str(e)}")
        else:
            st.info("⬆️ Please upload a CSV file to begin analysis")

    # Section 3: RESTORED Manual Model Configuration
    if "clinical_data" in st.session_state:
        data = st.session_state["clinical_data"] # Get data for column selection
        with st.expander("⚙️ Configure Analysis Parameters (Manual)", expanded=True):
            
            # --- Column Selection for Manual Run ---
            st.subheader("Column Mapping")
            cols = data.columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                time_col_selection = st.selectbox("Select Time-to-Event Column", options=cols, key="manual_time_col")
            with col2:
                event_col_selection = st.selectbox("Select Event Indicator Column", options=cols, key="manual_event_col")

            # --- KEY FIX: Save the user's selection to session_state ---
            # This was the missing piece that caused the 'no key "time_col"' error.
            st.session_state["time_col"] = time_col_selection
            st.session_state["event_col"] = event_col_selection
            # ----------------------------------------------------------------

            # --- Algorithm and Hyperparameter Selection ---
            st.subheader("Algorithm & Training Configuration")
            algo = st.selectbox("Select Survival Algorithm",
                              options=["CoxTime", "DeepSurv", "DeepHit", "TEXGISA"],
                              help="Select the model for manual execution.")

            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                batch_size = st.number_input("Batch Size", 32, 256, 64)
            with h_col2:
                epochs = st.number_input("Training Epochs", 100, 1000, 500)
            with h_col3:
                lr = st.number_input("Learning Rate", 1e-4, 1e-1, 1e-2, format="%.4f")
            
            config = {}
            if algo == "DeepHit":
                st.subheader("DeepHit Configuration")
                num_intervals = st.number_input("Time Intervals", 10, 100, 50)
                config["num_intervals"] = num_intervals
            
            # --- Manual Run Button ---
            if st.button("🚀 Start Survival Analysis", use_container_width=True, key="manual_run"):
                with st.spinner("🔬 Analyzing clinical patterns..."):
                    try:
                        results = run_analysis(algo, config, batch_size, epochs, lr)
                        st.session_state["results"] = results
                        st.success("🎉 Analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    # Section 4: Results Visualization
    if "results" in st.session_state:
        st.header("Clinical Insights")
        display_results(st.session_state["results"])

# This function supports the manual run button
def run_analysis(algo, config, batch_size, epochs, lr):
    data = st.session_state["clinical_data"]
    # It now safely reads the keys we saved above
    time_col = st.session_state["time_col"]
    event_col = st.session_state["event_col"]
    
    common_config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "time_col": time_col,
        "event_col": event_col
    }
    config.update(common_config)
    
    # This renaming is crucial for the backend functions
    data = data.rename(columns={time_col: 'duration', event_col: 'event'})
    
    if algo == "CoxTime":
        return coxtime.run_coxtime(data, config)
    elif algo == "DeepSurv":
        return deepsurv.run_deepsurv(data, config)
    elif algo == "DeepHit":
        return deephit.run_deephit(data, config)
    elif algo == "TEXGISA": # <-- 添加对新模型的调用
        # Pass the UI config to your new function
        config.update({"batch_size": batch_size, "epochs": epochs, "lr": lr})
        return run_texgisa(data, config)

# This function displays results from the manual run
def display_results(results):
    st.subheader("Validation Metrics")
    metrics_data = {"Metric": [], "Value": []}
    if results:
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                metrics_data["Metric"].append(key)
                metrics_data["Value"].append(value)
    
    metrics = pd.DataFrame(metrics_data)
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    
    if "Surv_Test" in results and isinstance(results["Surv_Test"], pd.DataFrame):
        st.subheader("Predicted Survival Trajectories")
        fig, ax = plt.subplots(figsize=(10, 6))
        for pid in results["Surv_Test"].columns[:5]:
            ax.step(results["Surv_Test"].index, results["Surv_Test"][pid], where="post", label=f"Patient {pid}")
        ax.set_xlabel("Follow-up Time")
        ax.set_ylabel("Survival Probability")
        ax.legend()
        st.pyplot(fig)