"""BERT在线预测器：加载微调最佳模型与元数据，对API和脚本提供统一的单条/批量预测能力。"""

import sys                                                        # 用于注入项目根目录
import json                                                       # 读取模型元数据
from pathlib import Path                                          # 跨平台路径处理

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # 必须在 src 导入之前注入项目根目录

import torch                                                      # 推理与设备管理
from transformers import (                                        # Hugging Face 模型与分词器
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.config import Config                                     # 全项目统一配置（只导出类，需自行实例化）


class BertNewsPredictor:

    def __init__(self):
        config = Config()                                          # 实例化全局配置
        self.device = config.device                                # 推理设备（GPU优先，否则CPU）

        metadata_path = config.bert_best_model_dir / "metadata.json"
        if not metadata_path.is_file():                            # 模型未训练时给出明确提示，避免晦涩报错
            raise FileNotFoundError(f"未找到最佳模型：{metadata_path}，请先运行 python src\\bert\\train.py")

        # 1. 读取模型元数据（类别列表 + 训练时的截断长度，保证推理与训练口径一致）
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        self.class_list = metadata["class_list"]
        self.max_length = metadata["max_length"]

        # 2. 加载Tokenizer与最佳模型（均从 best_model 目录读取，训练时已一起保存）
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_best_model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.bert_best_model_dir)
        self.model.to(self.device)                                 # 模型移动到设备
        self.model.eval()                                          # 设置评估模式（关闭dropout）

    def predict_one(self, text):
        """单条预测：返回文本、类别名与置信度。"""
        text = text.strip()
        if not text:
            raise ValueError("text 不能为空")

        # padding=False：单条无需补齐；截断长度与训练保持一致
        encoded = self.tokenizer(text, padding=False, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():                               # 推理阶段不计算梯度
            probabilities = torch.softmax(self.model(**encoded).logits, dim=1)   # logits 转概率
            score, pred_idx = torch.max(probabilities, dim=1)      # 最高概率对应预测类别

        return {"text": text, "pred_class": self.class_list[pred_idx.item()], "score": round(score.item(), 6)}

    def predict_batch(self, texts):
        """批量预测：一次 Tokenize + 一次前向推理，padding 在 Batch 内动态补齐。"""
        texts = [text.strip() for text in texts]
        if any(not text for text in texts):
            raise ValueError("texts 中不能包含空文本")

        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=1)   # 每条样本的类别概率
            scores, pred_indices = torch.max(probabilities, dim=1)               # 每条文本最大概率及类别

        return [
            {"text": text, "pred_class": self.class_list[idx], "score": round(score, 6)}
            for text, idx, score in zip(texts, pred_indices.tolist(), scores.tolist())
        ]


if __name__ == "__main__":
    # 自检：加载最佳模型并对示例文本做单条 + 批量预测（需先完成训练）
    predictor = BertNewsPredictor()
    print("单条预测：", predictor.predict_one("中华女子学院：本科层次仅1专业招男生"))
    batch = predictor.predict_batch(["英超曼城3比1击败阿森纳", "国产大模型多项评测超越国际同行"])
    for item in batch:
        print("批量预测：", item)
