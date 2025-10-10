import streamlit as st

def show():
    # 自定义CSS样式
    st.markdown("""
    <style>
    /* 全局容器样式 */
    .user-center {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    /* Radio组件容器 */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
        gap: 0.5rem !important;
    }

    /* 隐藏原生radio按钮 */
    div[role="radiogroup"] > div[data-testid="stVerticalBlock"] > div[role="radio"] {
        display: none !important;
    }

    /* 标签基础样式 */
    div[role="radiogroup"] label {
        display: block !important;
        margin: 0 !important;
        padding: 1rem !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        background: #ffffff !important;
        color: #000000 !important;
    }

    /* 悬停状态 */
    div[role="radiogroup"] label:hover {
        background: #f8f9fa !important;
        transform: translateX(5px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 选中状态 */
    div[role="radiogroup"] div[data-testid="stVerticalBlock"] > div[role="radio"][aria-checked="true"] + label {
        background: #2E86C1 !important;
        color: #ffffff !important;
        border-color: #2E86C1 !important;
    }

    /* 预览区样式 */
    .preview-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        min-height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 用户中心头部
    st.markdown("""
    <div class="user-center">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="width: 48px; height: 48px; border-radius: 50%; 
                        background: #2E86C1; display: grid; place-items: center; color: white;">
                U
            </div>
            <div>
                <h3 style="margin: 0; color: #2E86C1;">Researcher</h3>
                <p style="margin: 0; color: #666;">researcher@dhailab.com</p>
            </div>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #666;">Storage</span>
                <span style="color: #2E86C1;">65/100 GB</span>
            </div>
            <div style="height: 8px; background: #f0f2f6; border-radius: 4px;">
                <div style="width: 65%; height: 100%; background: #2E86C1; border-radius: 4px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 导航选项
    options = {
        "My Datasets": "📁",
        "My Experiments": "🧪",
        "My Results": "📊",
        "History": "🕒",
        "Favorite Models": "⭐"
    }
    
    selected = st.radio(
        "Navigation",
        [f"{icon} {name}" for name, icon in options.items()],
        label_visibility="collapsed"
    )
    selected_option = selected.split(" ", 1)[1]

    # 预览内容
    content_map = {
        "My Datasets": "🔍 Showing last 5 datasets...",
        "My Experiments": "⏳ Loading recent experiments...",
        "My Results": "📈 Visualizing model metrics...",
        "History": "🕰️ Retrieving 30-day activity...",
        "Favorite Models": "❤️ Loading saved models..."
    }

    st.markdown("""
    <div class="preview-container">
        <h4 style="color: #2E86C1; margin-top: 0;">Preview</h4>
        <div style="background: white; padding: 1rem; border-radius: 8px;">
            %s
        </div>
    </div>
    """ % content_map[selected_option], unsafe_allow_html=True)

if __name__ == "__main__":
    show()