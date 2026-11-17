import os
os.environ['PYCOX_DATA_DIR'] = os.path.join(os.getcwd(), 'pycox_data')
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import streamlit as st
import torch

# Keep small interactive training runs responsive on shared CPU hosts.
torch.set_num_threads(min(2, os.cpu_count() or 1))

# Streamlit 要求 set_page_config 是脚本中的第一条命令，因此放在任何其他 st 调用之前
st.set_page_config(page_title="DHAI Lab Survival Analysis Platform", layout="wide")

from pages_logic import home, algorithms, run_models, publications, contact, chat_with_agent
from utils import user_center
import pandas as pd
from sa_data_manager import DataManager # 确保导入了DataManager类

if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
def set_page(page):
    st.session_state["current_page"] = page

def main():
    # 自定义 CSS 样式，让导航按钮更美观
    st.markdown("""
    <style>
    .nav-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 20px;
        margin: 4px 2px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
    }
    .nav-button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 使用水平布局显示导航按钮
    nav = [
        ("🏠 Home", "Home"),
        ("📘 Algorithm Tutor", "Algorithms"),
        ("🧪 Model Training", "Run Models"),
        ("🤖 Chat with Agent", "Chat with Agent"),
        ("📄 Publications", "Publications"),
        ("✉️ Contact", "Contact"),
    ]
    cols = st.columns(len(nav))
    
    # 如果 session_state 中没有当前页面，默认设置为 "Home"
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # 在每个列中放置一个按钮，点击后调用 set_page() 修改当前页面
    for i, (label, value) in enumerate(nav):
        if cols[i].button(label, key=value, help=f"Go to {value} page", on_click=lambda v=value: set_page(v)):
            pass

    # 根据当前页面状态显示对应页面内容
    current_page = st.session_state.get("current_page", "Home")
    # label_map = dict(nav)
    # st.caption(f"You are here: {label_map.get(current_page, current_page)}")
        
    if current_page == "Home":
        home.show()
    elif current_page == "Algorithms":
        algorithms.show()
    elif current_page == "Run Models":
        run_models.show()
    elif current_page == "Publications":
        publications.show()
    elif current_page == "Chat with Agent":
        chat_with_agent.show()    
    elif current_page == "Contact":
        contact.show()

    # 侧边栏只显示用户中心和设置，不再重复显示主导航
    with st.sidebar:
        st.title("User Center")
        sidebar_options = ["Home", "Our Datasets", "Our Experiments", "Our Results", "History", "Favorite Models"]
        sidebar_selected = st.selectbox("Navigation", sidebar_options)
        if sidebar_selected != "Home":
            user_center.show(sidebar_selected)
        else:
            st.info("Return to the main page using the top navigation buttons.")
        
        st.title("Settings")
        st.markdown("Theme Switch: **Under Development**")
        st.markdown("Multi-language Support: **Under Development**")
        st.info("Guidance: Use the top navigation buttons to switch pages.")
        st.markdown("---")
        st.subheader("❓ Help Bot")

        # session for help chat
        if "help_messages" not in st.session_state:
            st.session_state.help_messages = []

        # render history (简化气泡)
        for role, content in st.session_state.help_messages[-6:]:  # 只显示最近几条
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Bot:** {content}")

        with st.form("help_form", clear_on_submit=True):
            help_q = st.text_input("Ask about the system or algorithms", "")
            ask = st.form_submit_button("Ask")
        if ask and help_q.strip():
            st.session_state.help_messages.append(("user", help_q.strip()))
            from utils.help_bot import answer
            ans = answer(help_q)
            st.session_state.help_messages.append(("assistant", ans))
            st.rerun()


if __name__ == "__main__":
    main()


