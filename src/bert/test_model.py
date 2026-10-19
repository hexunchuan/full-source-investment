"""BERT测试集评估：加载最佳微调模型，对 test.txt 全量评估，输出 Accuracy / Macro-F1 与逐类分类报告。"""

import sys                                                        # 用于注入项目根目录
from pathlib import Path                                          # 跨平台路径处理

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # 必须在 src 导入之前注入项目根目录

from transformers import (                                        # Hugging Face 模型与分词器
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.config import Config                                     # 全项目统一配置
from src.bert.data import load_class_list, build_dataloader       # BERT 数据链路
from src.bert.train import evaluate                               # 复用训练脚本的评估函数，保证指标口径一致

config = Config()


def test():
    class_list = load_class_list()                                # 加载类别列表

    model_dir = config.bert_best_model_dir                        # 统一Config中的BERT最佳模型目录
    print(f"Loading best model from {model_dir}")
    if not (model_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"未找到最佳模型：{model_dir}，请先运行 python src\\bert\\train.py")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)          # 加载最佳模型配套Tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)   # 加载最佳模型
    model.to(config.device)                                       # 移动到GPU/CPU

    test_loader = build_dataloader(config.test_path, tokenizer, shuffle=False)   # 测试集不打乱
    test_metrics = evaluate(model, test_loader, config.device, class_list)       # 全量评估

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")       # 整体准确率
    print(f"Test Macro-F1: {test_metrics['f1']:.4f}")             # 宏平均F1
    print(test_metrics["report"])                                 # 逐类详细分类报告


if __name__ == "__main__":
    test()