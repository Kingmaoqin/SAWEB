import streamlit as st
# Create DataManager once per session (safe if repeated)
if "data_manager" not in st.session_state:
    from sa_data_manager import DataManager
    st.session_state.data_manager = DataManager()

def show():
    st.title("DHAI Lab Survival Analysis Platform")
    st.caption("You are here: Home")

    with st.expander("🚀 How to Use This System (Quick Start)", expanded=True):
        st.markdown("""
    1.  **Upload Data**: Go to the Model Training page → Upload your CSV file (which must include `duration`, `event`, and feature columns).
    2.  **Map Columns**: On the page, map the time column to `duration` and the event column to `event`. The remaining columns will be automatically used as features.
    3.  **Select Algorithm**: Choose from the supported models: CoxTime, DeepSurv, DeepHit, or **Interpretable and Interactive Deep Survival Analysis with Time-dependent EXtreme Gradient Integration (TEXGISA)**.
    4.  **Quick Preview** (Recommended): Select **TEXGISA** → Click **👀 Preview FI** to first calculate the TEXGI feature importance.
    5.  **Set Priors**: Edit the rules in the *Expert Rules* table (e.g., `age >= mean, sign=+1`) and adjust the **λ_expert** value.
    6.  **Train with Priors**: Click **🚀 Train with Expert Priors**; compare the C-index and Feature Importance to see if they meet your expectations.
    7.  **View Charts**: In the *Charts* section, switch between the **Predicted Survival** and **Kaplan–Meier** plots.
        """)    
    # Survival Analysis Introduction Section
    st.subheader("What is Survival Analysis?")
    st.markdown("""
    **Survival Analysis** is a branch of statistics that analyzes the expected duration until one or more events occur, 
    widely used in medical research, engineering, economics, and more. Our specialized platform enables:

    - 🧠 **Interactive Education**: Learn survival analysis concepts through visual demonstrations
    - 📊 **Data Exploration**: Upload, visualize, and preprocess time-to-event datasets
    - ⚙️ **Algorithm Execution**: Run state-of-the-art survival models with configurable parameters
    - 📈 **Result Interpretation**: Understand outputs through explainable AI visualization

    This app serves as both a research tool and educational platform for survival analytics.
    """)
    
    # Lab Focus Section
    st.subheader("Research Focus Areas of DHAI Lab")
    
    # Create expandable sections for each research topic
    with st.expander("🔍 Interactive eXplainable AI (iXAI)"):
        st.markdown("""
        **Core Focus**: Developing AI systems that provide real-time, interactive explanations of their decision-making processes. 
        Combines causal reasoning with user feedback loops to create transparent predictive models.
        """)
        
    with st.expander("🤝 Human-Centered Interactive Algorithms"):
        st.markdown("""
        **Key Research**: Designing adaptive explanation systems that evolve with user expertise levels. 
        Focuses on multimodal explanation delivery (visual, textual, and interactive) for optimal human comprehension.
        """)

    with st.expander("🏥 Health/Biomedical Informatics"):
        st.markdown("""
        **Applications**: Developing predictive models for clinical outcomes analysis, treatment effect estimation, 
        and medical risk stratification using heterogeneous health data sources.
        """)

    with st.expander("📊 Data-Driven Decision Making (D3M)"):
        st.markdown("""
        **Methodology**: Creating framework-agnostic decision support systems that integrate domain knowledge with 
        machine learning outputs for clinical and operational decision optimization.
        """)

    with st.expander("🛡️ Trustworthy AI Systems"):
        st.markdown("""
        **Principles**: Engineering AI systems with built-in fairness, robustness, and accountability measures. 
        Special focus on privacy-preserving techniques for sensitive health data.
        """)

    with st.expander("🎯 Expertise-Grounded Causal Learning"):
        st.markdown("""
        **Innovation**: Hybrid modeling approaches that combine statistical causal inference with deep learning, 
        guided by domain expert knowledge and mechanistic understanding.
        """)

    with st.expander("🤖 AI/AGI-Agent Systems"):
        st.markdown("""
        **Vision**: Developing autonomous reasoning systems for complex survival analysis tasks. 
        Focus on multi-agent collaboration and human-AI teaming paradigms.
        """)

    # Platform Features
    st.subheader("Platform Capabilities")
    st.markdown("""
    **Interactive Survival Analysis Toolkit:**
    - 📚 **Educational Resources**: Curated tutorials with real-world case studies
    - 🧪 **Model Playground**: Compare traditional statistics vs. machine learning approaches
    - 🔄 **Data Pipelines**: Automated preprocessing for censored data and time-dependent covariates
    - 📉 **Visual Analytics**: Dynamic visualization of survival curves and feature impacts
    - ⚖️ **Model Auditing**: Fairness and robustness evaluation for survival models
    """)

    # Contact Section
    st.markdown("---")
    st.markdown("**Contact our team:** [xqin5@cougarnet.uh.edu](mailto:xqin5@cougarnet.uh.edu) | [Visit our website](https://)")