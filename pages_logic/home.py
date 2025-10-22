import streamlit as st
# Create DataManager once per session (safe if repeated)
if "data_manager" not in st.session_state:
    from sa_data_manager import DataManager
    st.session_state.data_manager = DataManager()

def show():
    st.title("Digital Health & Artificial Intelligence (DHAI) Lab Survival Analysis Platform")
    st.caption("You are here: Home")

    with st.expander("🚀 How to Use This System (Quick Start)", expanded=True):
        st.markdown("""
    **About this platform:** The Digital Health & Artificial Intelligence Lab Survival Analysis Platform guides clinicians through the full workflow of time-to-event modeling. Our flagship method, **TEXGISA – Interpretable and Interactive Deep Survival Analysis (TEXGISA)**, pairs transparent explanations with state-of-the-art predictive performance so medical teams can make confident, data-informed decisions.

    1.  **Upload your patient table**: Visit the *Model Training* or *Chat with Agent* page and upload a CSV file. It must contain a time-to-event column (e.g., follow-up months) and an event indicator (1 = event, 0 = censored).
    2.  **Confirm column names**: Map the time column to `duration` and the event column to `event`. All remaining numeric columns (lab values, vitals, therapies) will be used as model inputs automatically.
    3.  **Pick an algorithm**: Choose from CoxTime, DeepSurv, DeepHit, or **TEXGISA – Interpretable and Interactive Deep Survival Analysis**. CoxTime/DeepSurv/DeepHit follow familiar statistical or neural baselines, while TEXGISA produces clinician-friendly explanations.
    4.  **Decide on explanations**: On the Chat page you can launch TEXGISA with **feature importance** (full explanations for each variable) or **without feature importance** for a faster, metrics-only run.
    5.  **Review the results**: Check key metrics (e.g., concordance index), predicted survival curves, and the Kaplan–Meier chart. When feature importance is enabled, TEXGISA highlights which clinical factors drive earlier or later events.
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

    st.markdown("""
    **Clinician quick tips:**

    - Start with **TEXGISA with feature importance** to see which variables raise or lower risk over time.
    - Use the **Kaplan–Meier** plot to compare predicted survival against familiar empirical curves.
    - Export the tables from each page to share with your multidisciplinary team for decision support.
    """)
    
    # Lab Focus Section
    st.subheader("Research Focus Areas of the DHAI Lab")
    
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

    with st.expander("🤖 Artificial Intelligence / Artificial General Intelligence (AI/AGI) Agent Systems"):
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