import streamlit as st
import pandas as pd

def show():
    st.title("Publications")
    st.caption("You are here: Publications")

    st.markdown("Below are our latest research papers and datasets.")
    
    publications_df = pd.DataFrame({
        "Title": ["Interactive Explainable Deep Survival Analysis"],
        "Link": ["https://ieeexplore.ieee.org/abstract/document/10782608"]
    })
    
    st.dataframe(
        publications_df,
        column_config={
            "Link": st.column_config.LinkColumn(
                "Paper Link",  
                display_text="View Paper",  
                help="Click to view full paper"  
            )
        },
        hide_index=True,
        use_container_width=True
    )