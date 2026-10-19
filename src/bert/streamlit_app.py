"""Streamlit前端界面，左右分栏布局展示BERT新闻分类系统，通过HTTP请求调用localhost:8005的Flask预测API。"""

import streamlit as st
import requests
import time
import pandas as pd  # 批量结果表格与CSV导出

# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title="新闻文本分类系统（BERT）",
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

    /* 页签样式 */
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 42px; padding: 0 22px; border-radius: 12px 12px 0 0;
        background: #f2f7fb; font-weight: 600; color: #5d7488;
    }
    .stTabs [aria-selected="true"] {background: #92b4cc !important; color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)

# ==================== 类别展示配置（与 class.txt 顺序对应） ====================
CLASS_INFO = {
    "finance": ("💰", "财经"),
    "realty": ("🏠", "房产"),
    "stocks": ("📈", "股票"),
    "education": ("🎓", "教育"),
    "science": ("🔬", "科学"),
    "society": ("🏙️", "社会"),
    "politics": ("🏛️", "政治"),
    "sports": ("⚽", "体育"),
    "game": ("🎮", "游戏"),
    "entertainment": ("🎬", "娱乐"),
}

# ==================== 会话状态 ====================
if "predict_result" not in st.session_state:
    st.session_state.predict_result = None
if "predict_error" not in st.session_state:
    st.session_state.predict_error = None
if "history" not in st.session_state:
    st.session_state.history = []
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None  # 批量预测结果（DataFrame）
if "batch_meta" not in st.session_state:
    st.session_state.batch_meta = None  # 批量预测统计（条数/总耗时）
if "batch_error" not in st.session_state:
    st.session_state.batch_error = None

# 示例文本：点击按钮一键填入输入框
SAMPLE_TEXTS = [
    "中国人民银行宣布下调贷款市场报价利率",
    "英超曼城3比1击败阿森纳 哈兰德梅开二度",
    "国产大模型再升级 多项评测超越国际同行",
]


def fill_sample(text):
    """回调函数：把示例文本写入输入框（on_click 在重跑前执行）"""
    st.session_state.input_text = text


def fill_batch_sample(text):
    """回调函数：把批量示例写入批量输入框（on_click 在重跑前执行）"""
    st.session_state.batch_input_text = text


# ==================== 顶部横幅 ====================
st.markdown("""
<div class="hero-banner">
    <h1>📰 新闻文本分类系统</h1>
    <p>BERT（bert-base-chinese 微调） · 输入新闻标题，自动识别所属类别</p>
</div>
""", unsafe_allow_html=True)

# ==================== 页签：单条预测 / 批量预测 ====================
tab_single, tab_batch = st.tabs(["📝 单条预测", "📦 批量预测"])

# ==================== 页签一：左右分栏（左输入 / 右结果） ====================
with tab_single:
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
                url = 'http://localhost:8005/predict'  # BERT Flask API 专用端口 8005
                start_time = time.time()

                try:
                    # 发送 POST 请求（BERT 推理慢于 FastText，30 秒超时更稳妥）
                    with st.spinner("预测中..."):
                        response = requests.post(url, json=data, timeout=30)
                    elapsed_time = (time.time() - start_time) * 1000

                    # 后端返回结构：{"code": 0, "message": "success", "data": {"label": ..., "probability": ..., "elapsed_ms": ...}}
                    if response.status_code == 200:
                        result = response.json()
                        pred_class = result['data']['label']
                        probability = result['data']['probability']
                        infer_ms = result['data'].get('elapsed_ms')          # 后端纯推理耗时（毫秒）
                        emoji, zh_name = CLASS_INFO.get(pred_class, ("🏷️", pred_class))
                        st.session_state.predict_result = {
                            "class": pred_class,
                            "display": f"{emoji} {zh_name}",
                            "probability": probability,
                            "ms": elapsed_time,
                            "infer_ms": infer_ms,
                            "text": text_input,
                        }
                        st.session_state.predict_error = None

                        # 写入最近预测历史（只保留最新5条）
                        brief = text_input[:18] + ("…" if len(text_input) > 18 else "")
                        infer_str = f"{infer_ms:.0f} ms" if infer_ms is not None else f"{elapsed_time:.0f} ms"
                        st.session_state.history.append(
                            {"文本": brief, "类别": zh_name, "置信度": f"{probability:.2%}",
                             "推理耗时": infer_str}
                        )
                        st.session_state.history = st.session_state.history[-5:]
                    else:
                        # 后端可能返回 JSON 错误，也可能返回 HTML 错误页，需分别处理
                        try:
                            error_msg = response.json().get('message', f"HTTP {response.status_code}")
                        except ValueError:
                            error_msg = f"HTTP {response.status_code}"
                        st.session_state.predict_error = error_msg
                        st.session_state.predict_result = None

                except requests.exceptions.Timeout:
                    st.session_state.predict_error = "请求超时（30秒），请检查 Flask API 服务是否正常"
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
                col_m1.metric("模型推理耗时", f"{result['infer_ms']:.0f} ms"
                              if result.get('infer_ms') is not None else "—")
                col_m2.metric("请求总耗时", f"{result['ms']:.0f} ms")
                col_m3, col_m4 = st.columns(2)
                col_m3.metric("置信度", f"{result['probability']:.2%}")
                col_m4.metric("文本长度", f"{len(result['text'])} 字")

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

# ==================== 页签二：批量预测（左输入 / 右结果） ====================
BATCH_SAMPLE_TEXT = (
    "中国人民银行宣布下调贷款市场报价利率\n"
    "英超曼城3比1击败阿森纳 哈兰德梅开二度\n"
    "国产大模型再升级 多项评测超越国际同行"
)

with tab_batch:
    col_b_left, col_b_right = st.columns([3, 2], gap="large")

    # ---------- 左侧：批量输入区 ----------
    with col_b_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">✍️ 批量输入（每行一条文本）</div>', unsafe_allow_html=True)
            batch_input = st.text_area(
                "批量新闻文本",
                value="",
                height=220,
                label_visibility="collapsed",
                placeholder="每行输入一条要预测的中文文本，也可以通过下方上传 .txt 文件...",
                key="batch_input_text",
            )

            uploaded_file = st.file_uploader("或上传 .txt 文件（每行一条文本）", type=["txt"])
            if uploaded_file is not None:
                batch_input = uploaded_file.read().decode("utf-8", errors="ignore")
                st.caption(f"已读取文件：{uploaded_file.name}")

            if st.button("🚀 开始批量预测", use_container_width=True, key="batch_predict"):
                texts = [line.strip() for line in batch_input.splitlines() if line.strip()]
                st.session_state.batch_error = None
                st.session_state.batch_result = None
                st.session_state.batch_meta = None

                if not texts:
                    st.session_state.batch_error = "请输入至少一条非空文本（每行一条）"
                else:
                    url = 'http://localhost:8005/predict_batch'  # BERT Flask API 批量接口
                    chunk_size = 16                              # 每批请求的文本条数，控制进度粒度
                    total = len(texts)
                    start_time = time.time()

                    progress_bar = st.progress(0.0, text=f"准备预测 {total} 条文本...")
                    status_text = st.empty()

                    all_items = []
                    total_infer_ms = 0.0
                    has_infer_ms = False
                    try:
                        for start in range(0, total, chunk_size):
                            chunk = texts[start:start + chunk_size]
                            response = requests.post(url, json={'texts': chunk}, timeout=120)

                            if response.status_code != 200:
                                try:
                                    error_msg = response.json().get('message', f"HTTP {response.status_code}")
                                except ValueError:
                                    error_msg = f"HTTP {response.status_code}"
                                st.session_state.batch_error = error_msg
                                progress_bar.empty()
                                status_text.empty()
                                break

                            data = response.json()['data']
                            items = data['items'] if isinstance(data, dict) else data
                            all_items.extend(items)
                            if isinstance(data, dict) and data.get('elapsed_ms') is not None:
                                total_infer_ms += data['elapsed_ms']
                                has_infer_ms = True

                            done = len(all_items)
                            progress_bar.progress(min(done / total, 1.0),
                                                  text=f"已完成 {done}/{total} 条（{done / total * 100:.0f}%）")
                        else:
                            # 全部批次成功，写入结果
                            elapsed_time = (time.time() - start_time) * 1000
                            progress_bar.progress(1.0, text=f"✅ 全部 {total} 条预测完成，正在整理结果...")
                            status_text.success(f"预测完成，共 {total} 条")

                            rows = []
                            for i, item in enumerate(all_items, start=1):
                                emoji, zh_name = CLASS_INFO.get(item['label'], ("🏷️", item['label']))
                                rows.append({
                                    "序号": i,
                                    "文本": item['text'][:30] + ("…" if len(item['text']) > 30 else ""),
                                    "类别": f"{emoji} {zh_name}",
                                    "类别代码": item['label'],
                                    "置信度": f"{item['probability']:.2%}",
                                })
                            st.session_state.batch_result = pd.DataFrame(rows)
                            st.session_state.batch_meta = {
                                "count": len(rows),
                                "infer_ms": total_infer_ms if has_infer_ms else None,
                                "avg_infer_ms": (total_infer_ms / len(rows)) if has_infer_ms and rows else None,
                                "ms": elapsed_time,
                                "avg_ms": elapsed_time / len(rows) if rows else 0,
                            }
                            time.sleep(0.6)                              # 让 100% 状态短暂可见
                            progress_bar.empty()
                            status_text.empty()

                    except requests.exceptions.Timeout:
                        st.session_state.batch_error = "请求超时（120秒），请减少文本数量或检查 Flask API 服务"
                        progress_bar.empty()
                        status_text.empty()
                    except Exception as e:
                        st.session_state.batch_error = f"请求出错：{str(e)}"
                        progress_bar.empty()
                        status_text.empty()

            st.markdown("**🎲 快速试试批量示例：**")
            st.button("载入3条示例文本", key="batch_sample", use_container_width=True,
                      on_click=fill_batch_sample, args=(BATCH_SAMPLE_TEXT,))

    # ---------- 右侧：批量结果区 ----------
    with col_b_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">🎯 批量预测结果</div>', unsafe_allow_html=True)

            if st.session_state.batch_error:
                st.error(f"❌ {st.session_state.batch_error}")

            if st.session_state.batch_result is not None:
                meta = st.session_state.batch_meta
                col_bm1, col_bm2 = st.columns(2)
                col_bm1.metric("预测条数", f"{meta['count']} 条")
                col_bm2.metric("请求总耗时", f"{meta['ms']:.0f} ms")
                col_bm3, col_bm4 = st.columns(2)
                col_bm3.metric("推理总耗时", f"{meta['infer_ms']:.0f} ms" if meta.get('infer_ms') is not None else "—")
                col_bm4.metric("平均推理/条", f"{meta['avg_infer_ms']:.1f} ms"
                               if meta.get('avg_infer_ms') is not None else f"{meta['avg_ms']:.0f} ms")
                st.caption("推理总耗时为后端纯模型推理时间，请求总耗时含 HTTP 网络往返；批量一次前向推理远快于循环单条")

                st.dataframe(
                    st.session_state.batch_result,
                    hide_index=True,
                    use_container_width=True,
                    height=320,
                )
                csv_bytes = st.session_state.batch_result.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 下载预测结果 CSV",
                    data=csv_bytes,
                    file_name="bert_batch_predict.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            elif not st.session_state.batch_error:
                st.markdown(
                    '<div class="placeholder">在左侧输入多行文本或上传 .txt 文件<br>'
                    '点击「开始批量预测」<br>结果表格将显示在这里</div>',
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
    1. 确保 **Flask API** 已在 `localhost:8005` 运行
    2. 输入新闻标题或任意中文文本
    3. 点击「开始预测」查看类别、置信度与耗时
    """)

    st.divider()
    st.header("🗂️ 支持的类别")
    for name, (emoji, zh) in CLASS_INFO.items():
        st.markdown(
            f'<div class="cat-row">{emoji} <b>{zh}</b> · <code>{name}</code></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("模型：BERT（bert-base-chinese 微调） · 服务：Flask(8005) + Streamlit")

# ==================== 页脚提示 ====================
st.caption("💡 首次请求较慢（模型预热 + BERT 推理开销）属正常现象，后续请求耗时显著下降")
