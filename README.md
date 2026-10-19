# 新闻文本分类系统（FullSorceInvestment）

一个中文新闻文本多分类项目，实现了从数据探索（EDA）、数据预处理、模型训练、批量预测与测试集评估，到 **Flask API 服务** 和 **Streamlit Web 界面** 的完整机器学习闭环。

目前已实现三种分类方案：

1. 🌲 **TF-IDF + 随机森林（RF）**
2. ⚡ **FastText（词级 / 字符级 + AutoTune）**
3. 🔮 **BERT（bert-base-chinese 微调，Macro-F1 94.1%）**

用户只需在网页中输入一段新闻标题/文本，系统即可自动返回其所属类别（共 10 类）。

---

## ✨ 功能特性

- 📊 **探索性数据分析（EDA）**：统计 train/dev/test 三个数据集的样本量、类别分布、文本长度（均值/标准差/P50/P90/P95/P99），检查空文本、完全重复、同文异标、疑似乱码以及数据集之间的文本泄漏，输出两张图表与 JSON 报告至 `artifacts/eda/`。
- 🧹 **数据预处理（RF 链路）**：HTML 标签清洗、换行/制表符/零宽字符/乱码替换符处理、连续空白合并、jieba 中文分词、按 30 词截断，并对结果做字段、标签范围、截断长度等合法性校验，生成 `text / label / words` 三列的标准 CSV 至 `data/rf_processed/`。
- 🧹 **数据预处理（FastText 链路）**：`src/fasttext/preprocess.py` 直接读取 `data/raw/` 原始 TXT，生成 `__label__类别名 文本` 格式的字符级与词级训练文件至 `data/fasttext_processed/`。
- 🌲 **RF 模型训练与评估**：TF-IDF 向量化（结合停用词表过滤）+ 随机森林分类器（全 CPU 并行、固定随机种子 42）。训练集随机采样 50000 条，验证集 `dev.csv` 仅 transform 不 fit，输出准确率、精确率、召回率、F1（micro 口径）。
- ⚡ **FastText 模型训练与评估**：词级 / 字符级 Baseline 训练（`train_baseline.py`）、自动调参（`train_autotune.py`）、测试集完整评估（`evaluate.py`，输出 Accuracy 与逐类分类报告）。
- 🔮 **BERT 微调训练与评估**：加载本地 `bert-base-chinese` 预训练模型 + 分类头，支持 GPU/CPU 自动切换、Warmup + 线性衰减学习率、梯度裁剪、Early Stopping；每 Epoch 在 dev 集评估 Macro-F1，自动保存最佳模型与元数据；测试集全量评估输出 Accuracy / Macro-F1 与逐类分类报告。
- 🌐 **在线服务**：每个模型独立提供 Flask REST API 与 Streamlit 前端，支持类别中文名/emoji 展示、示例文本一键填入、请求耗时统计、最近预测历史与清空、错误提示（含超时保护）。BERT 前端额外支持 **批量预测**（多行文本 / 上传 .txt 文件、进度条、结果表格与 CSV 下载）。

## 🗂️ 支持的类别（共 10 类）

| 编号 | 类别 | 中文 | 编号 | 类别 | 中文 |
| --- | --- | --- | --- | --- | --- |
| 0 | finance | 💰 财经 | 5 | society | 🏙️ 社会 |
| 1 | realty | 🏠 房产 | 6 | politics | 🏛️ 政治 |
| 2 | stocks | 📈 股票 | 7 | sports | ⚽ 体育 |
| 3 | education | 🎓 教育 | 8 | game | 🎮 游戏 |
| 4 | science | 🔬 科学 | 9 | entertainment | 🎬 娱乐 |

---

## 🧱 技术栈

| 类别 | 工具/库 |
| --- | --- |
| 语言 | Python 3.11 |
| 数据处理 | pandas、numpy |
| 中文分词 | jieba |
| 机器学习 | scikit-learn（TfidfVectorizer、RandomForestClassifier） |
| 文本分类 | fasttext（词级 / 字符级） |
| 深度学习 | PyTorch、Hugging Face Transformers（bert-base-chinese 微调） |
| 模型评估 | sklearn.metrics（accuracy / precision / recall / f1 / classification_report） |
| 数据可视化 | matplotlib、seaborn |
| 模型持久化 | pickle、safetensors、fasttext .bin |
| 后端 API | Flask |
| 前端界面 | Streamlit |
| HTTP 请求 | requests |

---

## 📁 项目结构

项目采用「公共层 + 模型级端到端自包含」架构：每个模型目录内包含从数据准备、训练、评估到 Web 服务的完整链路。

    FullSorceInvestment/
    ├── data/                          # 数据目录（不入库，需自行准备/生成）
    │   ├── raw/                       # 原始数据（sentence \t label，含 class.txt）
    │   ├── rf_processed/              # RF 链路数据（train/dev/test.csv + stopwords.txt）
    │   └── fasttext_processed/        # FastText 链路数据（char / word 两种粒度 TXT）
    ├── bert_pretrained/               # 本地 BERT 预训练模型（不入库，需自行下载，见「准备数据」）
    │   └── bert-base-chinese/         #   config.json / vocab.txt / pytorch_model.bin 等
    ├── models/                        # 训练产物（不入库）
    │   ├── rf/                        # rf_model.pkl、tfidf_vectorizer.pkl
    │   ├── fasttext/                  # fasttext_word_autotune.bin 等
    │   └── bert/best_model/           # BERT 微调最佳模型（safetensors + tokenizer + metadata）
    ├── logs/                          # 训练日志
    ├── artifacts/eda/                 # EDA 图表与 JSON 报告
    ├── src/
    │   ├── config.py                  # 集中式路径配置（Config 类，全项目共用）
    │   ├── eda.py                     # 探索性数据分析
    │   ├── rf/                        # 随机森林端到端模块
    │   │   ├── preprocess.py          #   数据清洗与预处理
    │   │   ├── train.py               #   训练与验证集评估
    │   │   ├── evaluate.py            #   批量预测 + 测试集评估
    │   │   ├── predictor.py           #   单条预测函数（供 API 调用）
    │   │   ├── api.py                 #   Flask API（端口 8001）
    │   │   └── streamlit_app.py       #   Streamlit 前端
    │   ├── fasttext/                  # FastText 端到端模块
    │   │   ├── preprocess.py          #   生成 char/word 训练数据
    │   │   ├── train_baseline.py      #   Baseline 训练
    │   │   ├── train_autotune.py      #   自动调参
    │   │   ├── evaluate.py            #   测试集评估
    │   │   ├── predictor.py           #   在线预测封装类
    │   │   ├── api.py                 #   Flask API（端口 8003）
    │   │   └── streamlit_app.py       #   Streamlit 前端
    │   └── bert/                      # BERT 端到端模块
    │       ├── data.py                #   Dataset / DataLoader / 批量 collate Tokenize
    │       ├── metrics.py             #   Accuracy / Macro-F1 指标计算
    │       ├── train.py               #   微调训练（Warmup+衰减、Early Stopping，保存最佳模型）
    │       ├── test_model.py          #   测试集全量评估（逐类分类报告）
    │       ├── predictor.py           #   在线预测封装类（模型 + Tokenizer + 元数据）
    │       ├── api.py                 #   Flask API（端口 8005）
    │       └── streamlit_app.py       #   Streamlit 前端（支持批量预测 / CSV 下载）
    ├── .gitignore                     # 忽略数据、模型、日志等大体积产物
    └── requirements.txt               # 依赖清单

---

## 🚀 快速开始

### 1. 安装依赖

    pip install -r requirements.txt

> ⚠️ Windows 下 `pip install fasttext` 可能编译失败，推荐使用 conda 安装：
> `conda install -c conda-forge fasttext`

### 2. 准备数据

将原始数据放入 `data/raw/` 目录，格式为 Tab 分隔的两列：`sentence`（文本）与 `label`（类别编号，0~9），即 `train.txt / dev.txt / test.txt`；类别名称按顺序写入 `data/raw/class.txt`（行号即编号）。

> 📦 出于仓库体积考虑，`data/`、`models/` 与 `bert_pretrained/` 均被 `.gitignore` 排除，clone 后请按本节准备数据、按下方命令重新训练。

### 2.1 准备 BERT 预训练模型（运行 BERT 链路必需）

BERT 微调依赖本地 `bert-base-chinese` 预训练模型，需自行下载放到 `bert_pretrained/bert-base-chinese/`：

    # 方式一：用 huggingface_hub（推荐）
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download('bert-base-chinese', local_dir='bert_pretrained/bert-base-chinese')"

    # 方式二：用 git lfs 克隆
    git lfs install
    git clone https://huggingface.co/google-bert/bert-base-chinese bert_pretrained/bert-base-chinese

> 若在国内网络下下载缓慢，可设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"`（PowerShell）后重试。
> 下载后该目录应包含 `config.json`、`vocab.txt`、`tokenizer.json`、`pytorch_model.bin` 等文件。

### 3. 完整流程（在项目根目录执行）

    # ① 数据探索分析（输出图表与 JSON 报告到 artifacts/eda/）
    python src\eda.py

    # ===== 随机森林链路 =====
    # ② 数据预处理（data/raw/*.txt → data/rf_processed/*.csv）
    python src\rf\preprocess.py

    # ③ 训练模型（采样 5 万条训练，dev 评估，生成 models/rf/ 下两个 pkl）
    python src\rf\train.py

    # ④ 批量预测与测试集评估
    python src\rf\evaluate.py

    # ===== FastText 链路 =====
    # ⑤ 生成 char/word 训练数据（data/raw → data/fasttext_processed/）
    python src\fasttext\preprocess.py

    # ⑥ 训练词级 Baseline（保存至 models/fasttext/fasttext_news.bin）
    python src\fasttext\train_baseline.py

    # ⑦ 自动调参训练（约 5 分钟，保存至 models/fasttext/fasttext_word_autotune.bin）
    python src\fasttext\train_autotune.py

    # ⑧ 测试集评估（加载 autotune 模型，Accuracy + 逐类分类报告）
    python src\fasttext\evaluate.py

    # ===== BERT 链路 =====
    # ⑨ 微调训练（加载 bert_pretrained，dev 集按 Macro-F1 保存最佳模型到 models/bert/best_model）
    python src\bert\train.py

    # ⑩ 测试集全量评估（加载最佳模型，输出 Accuracy / Macro-F1 与逐类分类报告）
    python src\bert\test_model.py

---

## 🌐 启动 Web 在线预测

采用前后端分离架构，每个模型需 **同时启动两个服务，且 Flask 先启动**。

### 随机森林（Flask 8001 + Streamlit）

    # 终端 1
    python src\rf\api.py
    # 终端 2
    python -m streamlit run src\rf\streamlit_app.py

### FastText（Flask 8003 + Streamlit）

    # 终端 1
    python src\fasttext\api.py
    # 终端 2
    python -m streamlit run src\fasttext\streamlit_app.py

### BERT（Flask 8005 + Streamlit）

    # 终端 1
    python src\bert\api.py
    # 终端 2
    python -m streamlit run src\bert\streamlit_app.py

> BERT 前端额外支持 **批量预测**：多行文本 / 上传 `.txt` 文件，带进度条、结果表格与 CSV 下载。

浏览器打开 Streamlit 终端提示的地址（默认 http://localhost:8501），输入文本并点击预测按钮即可查看分类结果与请求耗时。

> 首次请求较慢（约 2 秒）属正常现象，包含 jieba 词典加载与模型预热开销。

### API 直接调用示例

    curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d "{\"text\": \"体验2D巅峰 倚天屠龙记十大创新概览\"}"

返回：

    {"text": "体验2D巅峰 倚天屠龙记十大创新概览", "pred_class": "game"}

FastText 接口（8003）返回格式：

    {"code": 0, "message": "success", "data": {"label": "game", "probability": 0.99}}

---

## ⚙️ 配置说明

- 全部数据/模型路径统一由 `src/config.py` 的 `Config` 类管理，基于 `PROJECT_ROOT = Path(__file__).resolve().parents[1]` 构建绝对路径；各模型目录内的入口脚本通过 `sys.path.insert` 注入项目根后以 `from src.config import Config` 绝对导入。
- 关键超参数：RF 训练采样上限 50000（`src/rf/train.py`）、分词截断 `MAX_WORDS = 30`（`src/rf/preprocess.py`，在线预测 `predict_fun` 保持一致）、随机种子 42。
- 服务端口：RF Flask `8001`、FastText Flask `8003`；对应 `streamlit_app.py` 中的调用地址需同步修改。

---

## 📌 常见问题

| 现象 | 原因 | 解决办法 |
| --- | --- | --- |
| Streamlit 提示「请求出错 / 请求超时」 | Flask 未启动或端口不符 | 确认对应模型的 Flask 正在 8001 / 8003 运行 |
| FileNotFoundError | 相对路径 / 数据文件缺失 | 在**项目根目录**执行命令，检查 `data/raw/` 等目录文件是否存在 |
| ModuleNotFoundError: config / predictor | 直接以裸文件名运行脚本 | 按 README 命令从项目根运行，入口脚本会自动注入根路径 |
| fasttext 安装失败 | Windows 下 pip 编译问题 | 改用 `conda install -c conda-forge fasttext` |
| 端口被占用 | 8001/8003/8501 已被使用 | 关闭旧进程，或修改端口（`api.py` 与 `streamlit_app.py` 需同步修改） |
| 首次预测较慢 | jieba 词典加载与模型预热 | 属正常现象，后续请求耗时显著下降 |

---

## 🗺️ Roadmap

- [x] TF-IDF + 随机森林分类链路（含 Web 服务）
- [x] FastText 词级 / 字符级分类链路（含自动调参与 Web 服务）
- [x] BERT 微调分类链路（含 Web 服务与批量预测）
- [ ] 三模型统一口径横向对比（compare 报告）

---

## 📄 许可证

本项目仅用于学习与研究用途。
