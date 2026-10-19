"""加载TF-IDF向量化器和随机森林模型，在完全独立的dev.csv与test.csv上做最终评估（初步验证已在train.py内部划分子集完成）。"""

import sys                                                          # 导包
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # 将项目根目录加入模块搜索路径

import pandas as pd                                                 # 读取表格数据
import pickle                                                       # 加载向量化器和模型
from sklearn.metrics import (                                       # 准确率/精确率/召回率/F1值/每个类别的详细指标
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)
from src.config import Config                                       # 导入项目路径配置


conf = Config()  # 创建配置对象

# 加载随机森林模型
with open(conf.rf_model_save_path, "rb") as file:
    model = pickle.load(file)

# 加载TF-IDF向量化器（仅在训练子集上fit过，这里只用transform）
with open(conf.tfidf_model_save_path, "rb") as file:
    tfidf = pickle.load(file)

# 读取类别名称，用于分类报告
with open(conf.class_data_path, encoding="utf-8") as file:
    target_names = [line.strip() for line in file if line.strip()]


# todo 1. 对dev.csv做完整评估（train.py未使用dev，此处为完全未见过的数据）

dev_df = pd.read_csv(conf.process_dev_path)

# 注意：预测时只能使用transform，不能重新fit_transform
dev_features = tfidf.transform(dev_df["words"])

# 使用随机森林预测新闻类别
dev_df["predicted_label"] = model.predict(dev_features)

print("dev数据预测完成！")
print(dev_df[["text", "label", "predicted_label"]].head())

# 与 train.py 初步验证相同的四个指标口径（micro），方便对比内部验证与独立dev的差距
y_true_dev = dev_df["label"]
y_pred_dev = dev_df["predicted_label"]

print("\n=== 独立验证集(dev)整体指标 ===")
print(f"准确率：{accuracy_score(y_true_dev, y_pred_dev):.4f}")
print(f"精确率(micro)：{precision_score(y_true_dev, y_pred_dev, average='micro'):.4f}")
print(f"召回率(micro)：{recall_score(y_true_dev, y_pred_dev, average='micro'):.4f}")
print(f"F1-score(micro)：{f1_score(y_true_dev, y_pred_dev, average='micro'):.4f}")


# todo 2. 对test.csv做最终评估

test_df = pd.read_csv(conf.process_test_path)

# 同样只能 transform，避免污染训练阶段学到的词表
test_features = tfidf.transform(test_df["words"])

# 在测试集上完成预测
y_true = test_df["label"]
y_pred = model.predict(test_features)

# 将预测结果写回表格，便于打印时查看错分样本
test_df["predicted_label"] = y_pred

print("\ntest数据预测完成！")
print(f"测试集样本数：{len(test_df)}")
print(test_df[["text", "label", "predicted_label"]].head())

# 计算并输出四个整体评估指标，与 train.py 中的 micro 口径保持一致
print("\n=== 测试集最终整体指标 ===")
print(f"准确率：{accuracy_score(y_true, y_pred):.4f}")
print(f"精确率(micro)：{precision_score(y_true, y_pred, average='micro'):.4f}")
print(f"召回率(micro)：{recall_score(y_true, y_pred, average='micro'):.4f}")
print(f"F1-score(micro)：{f1_score(y_true, y_pred, average='micro'):.4f}")

# 打印每个类别的分类报告，显式指定labels避免某些类别缺失时target_names错位
print("\n=== 测试集分类报告 ===")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(target_names))),
    target_names=target_names,
    digits=4,
    zero_division=0  # 类别无样本时显示0而不是报错
))
