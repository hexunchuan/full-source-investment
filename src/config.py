"""定义Config配置类，集中管理全项目的数据、模型、输出文件路径与超参数，并在初始化时安全创建输出目录。

本文件是全项目唯一的配置入口：RF、FastText、BERT 三个模型的路径与超参数统一在此管理，
并按模块分区注释，供各模型目录内的入口脚本通过 `from src.config import Config` 绝对导入。
"""

from pathlib import Path
import os

# BERT 训练依赖 torch；RF / FastText 不需要 torch。
# 这里安全导入，避免在未安装 torch 的环境下破坏 RF / FastText 对本配置文件的引用。
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1] # 获取项目根目录，parents[1]代表上一级目录


class Config:
    """全项目共用的路径与超参数配置，初始化时自动创建缺失的输出目录。"""

    def __init__(self):

        # ==================================================================
        # 🔗 公共层配置（原始 TXT 数据，FastText 与 BERT 复用同一份原始集）
        # ==================================================================

        # 原始数据目录（TXT格式，全项目统一入口）
        self.raw_data_dir = PROJECT_ROOT / "data" / "raw"

        # 原始训练/验证/测试集路径（FastText 与 BERT 共用）
        self.train_path = str(self.raw_data_dir / "train.txt")
        self.dev_path = str(self.raw_data_dir / "dev.txt")
        self.test_path = str(self.raw_data_dir / "test.txt")

        # 类别名称文件（行号即类别编号，全项目共用）
        self.class_data_path = str(self.raw_data_dir / "class.txt")
        self.class_path = self.class_data_path  # 兼容别名

        # ==================================================================
        # 🌲 随机森林（RF）配置：预处理CSV + TF-IDF + RandomForest 模型产物
        # ==================================================================

        # RF 处理后数据目录（CSV格式）
        self.rf_processed_dir = PROJECT_ROOT / "data" / "rf_processed"

        # 停用词文件路径（RF 的 TF-IDF 过滤使用）
        self.stop_words_path = str(self.rf_processed_dir / "stopwords.txt")

        # RF 训练 / 验证 / 测试数据路径
        self.process_train_path = str(self.rf_processed_dir / "train.csv")
        self.process_dev_path = str(self.rf_processed_dir / "dev.csv")
        self.process_test_path = str(self.rf_processed_dir / "test.csv")

        # 随机森林模型保存路径
        self.rf_model_save_path = str(PROJECT_ROOT / "models" / "rf" / "rf_model.pkl")

        # TF-IDF向量化器保存路径
        self.tfidf_model_save_path = str(PROJECT_ROOT / "models" / "rf" / "tfidf_vectorizer.pkl")

        # ==================================================================
        # ⚡ FastText 配置：char/word 训练文本 + baseline/autotune 模型产物
        # ==================================================================

        # FastText 处理后数据目录（专用TXT格式）
        self.fasttext_processed_dir = PROJECT_ROOT / "data" / "fasttext_processed"

        # FastText 模型目录
        self.model_dir = PROJECT_ROOT / "models" / "fasttext"

        # Baseline 模型路径（默认超参数训练）
        self.fasttext_word_baseline_path = str(self.model_dir / "fasttext_word_baseline.bin")
        self.fasttext_char_baseline_path = str(self.model_dir / "fasttext_char_baseline.bin")

        # AutoTune 模型路径（自动调参训练）
        self.fasttext_word_autotune_path = str(self.model_dir / "fasttext_word_autotune.bin")
        self.fasttext_char_autotune_path = str(self.model_dir / "fasttext_char_autotune.bin")

        # 默认加载模型路径（兼容别名，API/评估统一加载词级 Baseline）
        self.fasttext_model_path = self.fasttext_word_baseline_path
        self.processed_data_dir = self.fasttext_processed_dir  # 兼容别名

        # ==================================================================
        # 🔮 BERT 配置：预训练模型目录 + 微调产物 + 训练超参数
        # ==================================================================

        # 本地预训练模型目录（bert-base-chinese，需自行下载至此）
        self.bert_pretrained_dir = PROJECT_ROOT / "bert_pretrained"
        self.bert_model_name = "bert-base-chinese"  # 模型文件夹名，from_pretrained 的实际加载层级
        self.pretrained_model_path = str(self.bert_pretrained_dir / self.bert_model_name)  # 供 from_pretrained 使用

        # BERT 模型输出目录与最佳模型保存目录
        self.bert_model_dir = PROJECT_ROOT / "models" / "bert"
        self.bert_best_model_dir = self.bert_model_dir / "best_model"

        # 自动选择 GPU 或 CPU（torch 缺失时兜底为 "cpu"，不影响 RF/FastText 使用本配置）
        # 必须放在超参数之前：batch_size / num_workers 依赖 is_gpu 判断运行环境
        if _TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.is_gpu = str(self.device).startswith("cuda")

        # --- BERT 训练超参数 ---
        self.batch_size = 128 if self.is_gpu else 16   # GPU 服务器按显存调：8G→32，16G→64，24G+→128
        self.max_length = 64              # 最大 Token 长度
        self.num_epochs = 5               # 最大训练轮数
        self.learning_rate = 3e-5         # BERT 微调常用较小学习率
        self.weight_decay = 0.01          # 权重衰减
        self.warmup_ratio = 0.1           # Warmup 比例
        self.patience = 2                 # Early Stopping 容忍轮数
        self.max_grad_norm = 1.0          # 梯度裁剪阈值
        self.seed = 42                    # 随机种子
        self.num_workers = 4 if self.is_gpu else 0     # GPU 服务器多进程加载；Windows/CPU 环境用 0

        # ==================================================================
        # 📁 安全创建模型/数据输出目录，避免训练后保存时报 FileNotFoundError
        # ==================================================================
        os.makedirs(os.path.dirname(self.rf_model_save_path), exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.fasttext_processed_dir, exist_ok=True)
        os.makedirs(self.bert_best_model_dir, exist_ok=True)


if __name__ == '__main__':
    # 测试1：打印项目根目录与训练设备，确认路径推导和 torch 检测正确
    conf = Config()
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"torch 可用：{_TORCH_AVAILABLE}，BERT 训练设备：{conf.device}")

    # 测试2：遍历全部路径配置，逐项打印并检查是否存在
    # 目录类属性（按模块分区）
    dir_attrs = [
        "raw_data_dir",                # 公共层：原始数据目录
        "rf_processed_dir",            # RF：处理后数据目录
        "fasttext_processed_dir",      # FastText：处理后数据目录
        "model_dir",                   # FastText：模型目录
        "bert_pretrained_dir",         # BERT：预训练模型目录
        "pretrained_model_path",       # BERT：from_pretrained 加载路径（目录）
        "bert_model_dir",              # BERT：模型输出目录
        "bert_best_model_dir",         # BERT：最佳模型目录
    ]

    # 文件类属性
    file_attrs = [
        "train_path",                  # 公共层：原始训练集
        "dev_path",                    # 公共层：原始验证集
        "test_path",                   # 公共层：原始测试集
        "class_data_path",             # 公共层：类别名称文件
        "stop_words_path",             # RF：停用词文件
        "process_train_path",          # RF：训练CSV
        "process_dev_path",            # RF：验证CSV
        "process_test_path",           # RF：测试CSV
        "rf_model_save_path",          # RF：模型文件
        "tfidf_model_save_path",       # RF：TF-IDF向量化器文件
        "fasttext_model_path",         # FastText：默认模型文件
        "fasttext_word_baseline_path",  # FastText：词级Baseline
        "fasttext_char_baseline_path",  # FastText：字级Baseline
        "fasttext_word_autotune_path",  # FastText：词级AutoTune
        "fasttext_char_autotune_path",  # FastText：字级AutoTune
    ]

    print("\n=== 目录检查 ===")
    missing_dirs = []
    for attr in dir_attrs:
        path = getattr(conf, attr)                    # 按属性名取值，统一处理
        exists = Path(path).is_dir()                  # 判断目录是否存在
        print(f"[{'OK' if exists else '缺失'}] {attr}: {path}")
        if not exists:
            missing_dirs.append(attr)

    print("\n=== 文件检查 ===")
    missing_files = []
    for attr in file_attrs:
        path = getattr(conf, attr)
        exists = Path(path).is_file()                 # 判断文件是否存在
        print(f"[{'OK' if exists else '缺失'}] {attr}: {path}")
        if not exists:
            missing_files.append(attr)

    # 测试3：验证兼容别名指向一致
    print("\n=== 别名一致性检查 ===")
    assert conf.class_path == conf.class_data_path, "class_path 别名不一致"
    assert conf.processed_data_dir == conf.fasttext_processed_dir, "processed_data_dir 别名不一致"
    assert conf.fasttext_model_path == conf.fasttext_word_baseline_path, "fasttext_model_path 别名不一致"
    assert conf.pretrained_model_path == str(conf.bert_pretrained_dir / conf.bert_model_name), "pretrained_model_path 与预训练目录不一致"
    assert str(conf.bert_best_model_dir).startswith(str(conf.bert_model_dir)), "bert_best_model_dir 不在 bert_model_dir 下"
    print("[OK] class_path == class_data_path")
    print("[OK] processed_data_dir == fasttext_processed_dir")
    print("[OK] fasttext_model_path == fasttext_word_baseline_path")
    print("[OK] pretrained_model_path == bert_pretrained_dir / bert_model_name")
    print("[OK] bert_best_model_dir 位于 bert_model_dir 之下")

    # 汇总测试结果
    print("\n=== 测试汇总 ===")
    print(f"目录缺失数：{len(missing_dirs)}{f'（{missing_dirs}）' if missing_dirs else ''}")
    print(f"文件缺失数：{len(missing_files)}{f'（{missing_files}）' if missing_files else ''}")
    if not missing_dirs and not missing_files:
        print("全部路径配置有效，可以开始训练/预测流程")
    else:
        print("存在缺失路径，请先运行 preprocess / train 生成对应产物，或检查原始数据是否就位")


