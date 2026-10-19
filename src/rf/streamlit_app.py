"""Streamlit前端界面，左右分栏布局展示RF新闻分类系统，通过HTTP请求调用localhost:8001的Flask预测API。"""

import streamlit as st
import requests
import time

# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title="新闻文本分类系统",
    page_icon="📰",
    layout="wide",
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {max-width: 1280px; padding-top: 1.5rem;}

    /* 顶部横幅：居中大标题 */
    .hero-banner {
        background: linear-gradient(120deg, #e3edf5 0%, #eef3ec 100%);
        border: 1px solid #d5e2ec;
        padding: 26px 26px 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 22px;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 3px;
        color: #2e4a61;
    }
    .hero-banner p {
        margin: 8px 0 0;
        font-size: 0.95rem;
        color: #7b93a5;
        letter-spacing: 1px;
    }

    /* 面板标题 */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #3c586e;
        margin-bottom: 4px;
    }

    /* 结果面板：大类别展示 */
    .result-big {
        font-size: 2rem; font-weight: 800; color: #3c586e;
        background: linear-gradient(120deg, #f2f7fb, #f7faf6);
        border: 1px solid #dce8f1; border-radius: 14px;
        padding: 20px 24px; margin-bottom: 6px;
    }

    /* 未预测时的占位区 */
    .placeholder {
        border: 2px dashed #dde5ec; border-radius: 14px;
        padding: 70px 20px; text-align: center;
        color: #a2b3c0; font-size: 0.95rem; line-height: 2.2;
    }

    /* 文本引用块 */
    .text-quote {
        border-left: 3px solid #b8cfe0; padding: 6px 14px;
        color: #5d7488; font-size: 0.88rem; background: #f6f9fb;
        border-radius: 0 8px 8px 0; margin-top: 10px;
    }

    /* 指标卡片 */
    [data-testid="stMetric"] {
        background: #fbfcfd; border: 1px solid #e4eaf0;
        border-radius: 12px; padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] {color: #8296a6;}
    [data-testid="stMetricValue"] {font-size: 1.35rem; color: #46606f;}

    /* 主按钮 */
    .stButton > button {
        width: 100%; font-weight: 600; padding: 8px 0;
        border-radius: 12px; border: none; color: #ffffff;
        background: #92b4cc;
        box-shadow: 0 2px 8px rgba(146, 180, 204, 0.35);
    }
    .stButton > button:hover {
        background: #7ea6c2;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(146, 180, 204, 0.45);
    }

    /* 卡片边框统一 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border-color: #e3eaf1 !important;
    }

    /* 左右两列内的卡片强制等高 */
    div[data-testid="stColumn"] div[data-testid="stVerticalBlock"] {height: 100%;}
    div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {height: 100%;}

    /* 侧边栏类别行 */
    .cat-row {padding: 5px 10px; border-radius: 8px; font-size: 0.9rem; margin-bottom: 2px;}
    .cat-row:hover {background: #f0f5f9;}
</style>
""", unsafe_allow_html=True)

# ==================== 类别展示配置（与 class.txt 顺序对应） ====================
CLASS_INFO = {
    "finance":       ("💰", "财经"),
    "realty":        ("🏠", "房产"),
    "stocks":        ("📈", "股票"),
    "education":     ("🎓", "教育"),
    "science":       ("🔬", "科学"),
    "society":       ("🏙️", "社会"),
    "politics":      ("🏛️", "政治"),
    "sports":        ("⚽", "体育"),
    "game":          ("🎮", "游戏"),
    "entertainment": ("🎬", "娱乐"),
}

# ==================== 会话状态 ====================
if "predict_result" not in st.session_state:
    st.session_state.predict_result = None
if "predict_error" not in st.session_state:
    st.session_state.predict_error = None
if "history" not in st.session_state:
    st.session_state.history = []

# 示例文本：点击按钮一键填入输入框
SAMPLE_TEXTS = [
    "中国人民银行宣布下调贷款市场报价利率",
    "英超曼城3比1击败阿森纳 哈兰德梅开二度",
    "国产大模型再升级 多项评测超越国际同行",
]

def fill_sample(text):
    """回调函数：把示例文本写入输入框（on_click 在重跑前执行）"""
    st.session_state.input_text = text

# ==================== 顶部横幅 ====================
st.markdown("""
<div class="hero-banner">
    <h1>📰 新闻文本分类系统</h1>
    <p>TF-IDF + 随机森林 · 输入新闻标题，自动识别所属类别</p>
</div>
""", unsafe_allow_html=True)

# ==================== 左右分栏：左输入 / 右结果 ====================
col_left, col_right = st.columns([3, 2], gap="large")

# ---------- 左侧：输入区 ----------
with col_left:
    with st.container(border=True):
        st.markdown('<div class="section-title">✍️ 输入文本</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "新闻标题或正文",
            value="中国人民公安大学2012年硕士研究生目录及书目",
            height=200,
            label_visibility="collapsed",
            placeholder="在此输入要预测的中文文本...",
            key="input_text",
        )

        if st.button("🚀 开始预测", use_container_width=True):
            data = {'text': text_input}
            url = 'http://localhost:8001/predict'
            start_time = time.time()

            try:
                # 发送 POST 请求（10秒超时，避免后端挂起时前端一直等待）
                with st.spinner("预测中..."):
                    response = requests.post(url, json=data, timeout=10)
                elapsed_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()
                    pred_class = result['pred_class']
                    emoji, zh_name = CLASS_INFO.get(pred_class, ("🏷️", pred_class))
                    st.session_state.predict_result = {
                        "class": pred_class,
                        "display": f"{emoji} {zh_name}",
                        "ms": elapsed_time,
                        "text": text_input,
                    }
                    st.session_state.predict_error = None

                    # 写入最近预测历史（只保留最新5条）
                    brief = text_input[:18] + ("…" if len(text_input) > 18 else "")
                    st.session_state.history.append(
                        {"文本": brief, "类别": zh_name, "耗时": f"{elapsed_time:.0f} ms"}
                    )
                    st.session_state.history = st.session_state.history[-5:]
                else:
                    # 后端可能返回 JSON 错误，也可能返回 HTML 错误页，需分别处理
                    try:
                        error_msg = response.json().get('error', f"HTTP {response.status_code}")
                    except ValueError:
                        error_msg = f"HTTP {response.status_code}"
                    st.session_state.predict_error = error_msg
                    st.session_state.predict_result = None

            except requests.exceptions.Timeout:
                st.session_state.predict_error = "请求超时（10秒），请检查 Flask API 服务是否正常"
                st.session_state.predict_result = None
            except Exception as e:
                st.session_state.predict_error = f"请求出错：{str(e)}"
                st.session_state.predict_result = None

        st.markdown("**🎲 快速试试这些文本：**")
        sample_cols = st.columns(len(SAMPLE_TEXTS))
        for i, (col, sample) in enumerate(zip(sample_cols, SAMPLE_TEXTS)):
            col.button(
                sample[:10] + "…",
                key=f"sample_{i}",
                use_container_width=True,
                on_click=fill_sample,
                args=(sample,),
                help=sample,
            )

# ---------- 右侧：结果区 ----------
with col_right:
    with st.container(border=True):
        st.markdown('<div class="section-title">🎯 预测结果</div>', unsafe_allow_html=True)

        if st.session_state.predict_error:
            st.error(f"❌ {st.session_state.predict_error}")

        result = st.session_state.predict_result
        if result:
            st.markdown(
                f'<div class="result-big">{result["display"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"类别代码：{result['class']}")

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("请求耗时", f"{result['ms']:.0f} ms")
            col_m2.metric("文本长度", f"{len(result['text'])} 字")

            st.markdown(
                f'<div class="text-quote">📝 {result["text"][:60]}'
                f'{"..." if len(result["text"]) > 60 else ""}</div>',
                unsafe_allow_html=True,
            )
        elif not st.session_state.predict_error:
            st.markdown(
                '<div class="placeholder">在左侧输入文本并点击「开始预测」<br>'
                '识别结果将显示在这里</div>',
                unsafe_allow_html=True,
            )

# ==================== 底部通栏：最近预测 ====================
with st.container(border=True):
    title_col, clear_col = st.columns([5, 1])
    with title_col:
        st.markdown('<div class="section-title">🕘 最近预测</div>', unsafe_allow_html=True)
    with clear_col:
        if st.session_state.history:
            if st.button("🗑️ 清空记录", key="clear_history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        hide_index=True,
        use_container_width=True,
        height=230,
    )
else:
    st.caption("暂无预测记录，完成一次预测后会显示在这里")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("ℹ️ 使用说明")
    st.markdown("""
    1. 确保 **Flask API** 已在 `localhost:8001` 运行
    2. 输入新闻标题或任意中文文本
    3. 点击「开始预测」查看类别与耗时
    """)

    st.divider()
    st.header("🗂️ 支持的类别")
    for name, (emoji, zh) in CLASS_INFO.items():
        st.markdown(
            f'<div class="cat-row">{emoji} <b>{zh}</b> · <code>{name}</code></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("模型：TF-IDF + RandomForest · 服务：Flask + Streamlit")

# ==================== 页脚提示 ====================
st.caption("💡 首次请求较慢属正常现象（jieba 词典加载与模型预热）")
