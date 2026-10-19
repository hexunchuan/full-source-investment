"""
只读取RF预处理训练集CSV，随机采样后按80/20内部划分训练子集与验证子集；
TF-IDF仅在训练子集上fit，训练随机森林模型后用验证子集做初步验证，
最后保存模型和向量化器（独立dev/test上的最终评估由evaluate.py完成）。
"""

import sys                                                          # 导包
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # 将项目根目录加入模块搜索路径，保证任意工作目录下都能 import src.xxx

import pandas as pd                                                 # 读取和处理表格数据
import pickle                                                       # 保存模型和向量化器
from sklearn.feature_extraction.text import TfidfVectorizer         # 将文本转换为TF-IDF特征
from sklearn.ensemble import RandomForestClassifier                 # 随机森林分类器
from sklearn.model_selection import train_test_split                # 训练集内部划分训练/验证子集
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 准确率/精确率/召回率/F1值
from src.config import Config                                       # 导入项目配置


# todo 1. Pandas的基础设置

pd.set_option("display.expand_frame_repr", False)  # 避免宽表格换行
pd.set_option("display.max_columns", None)  # 显示所有字段
conf = Config()  # 创建配置对象


# todo 2. 只读取训练集，并在内部划分为训练子集和验证子集

# 读取训练集原始数据，随机采样5万条（固定种子42保证可复现），采样后再做内部划分
train_df = pd.read_csv(conf.process_train_path).sample(n=50000, random_state=42).reset_index(drop=True)

# 按 80/20 划分：80% 作为训练子集用于模型学习，20% 作为验证子集做初步验证（不参与fit）
# 注意：这是训练集内部划分的子集，与独立文件 data/rf_processed/dev.csv 无关
# shuffle=True 先打乱再切分，random_state=42 保证每次划分结果一致
train_sub_df, dev_sub_df = train_test_split(
    train_df,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

# words是生成的分词结果，作为模型输入
x_train_words = train_sub_df["words"]
x_dev_words = dev_sub_df["words"]

# label是新闻类别编号，作为模型要学习的目标
y_train = train_sub_df["label"]
y_dev = dev_sub_df["label"]

print("训练数据前5行：")
print(train_df.head())
print(f"总样本数（采样后）：{len(train_df)}")
print(f"训练子集样本数：{len(train_sub_df)}")
print(f"验证子集样本数：{len(dev_sub_df)}")


# todo 3. 文本特征提取：TF-IDF向量化

# 读取停用词文件，并按照空白字符切分成列表
with open(conf.stop_words_path, encoding="utf-8") as f:
    stop_words = f.read().split()

# 查看停用词列表的开头和结尾
print(f"stop_words：\n{stop_words[:5]}, {stop_words[-5:]}")

# 创建TF-IDF向量化器，并指定停用词列表
tfidf = TfidfVectorizer(stop_words=stop_words)

# 只在训练子集上fit，学习词表和IDF权重
x_train = tfidf.fit_transform(x_train_words)

# 验证子集只transform，避免数据泄露
x_dev = tfidf.transform(x_dev_words)

# 查看特征矩阵形状
print(f"训练子集特征矩阵shape：{x_train.shape}")
print(f"验证子集特征矩阵shape：{x_dev.shape}")

# 查看TF-IDF词汇表的大小
print(f"词汇表大小：{len(tfidf.vocabulary_)}")


# todo 4. 模型训练和初步验证

# 创建随机森林分类器，使用全部CPU核心加速训练
model = RandomForestClassifier(n_jobs=-1, random_state=42)

# 只在训练子集（80%）上拟合模型
model.fit(x_train, y_train)

# 使用验证子集（内部划分的20%）完成预测（模型从未见过这部分数据），做初步验证
y_pred = model.predict(x_dev)

# 查看前10条预测结果
print(f"预测结果：\n{y_pred[:10]}")

# 计算并输出四个初步验证指标（最终评估请使用 evaluate.py 在独立dev/test集上进行）
# micro：全局统计所有类别的TP/FP/FN后计算指标，关注整体性能
# macro：每个类别独立计算指标后取算术平均，关注每类（尤其是稀有类）性能
# weighted：按类别样本数加权平均，偏向多数类
print("\n=== 验证子集（train内部20%）整体指标 ===")
print(f"准确率：{accuracy_score(y_dev, y_pred):.4f}")
print(f"精确率(micro)：{precision_score(y_dev, y_pred, average='micro'):.4f}")
print(f"召回率(micro)：{recall_score(y_dev, y_pred, average='micro'):.4f}")
print(f"F1-score(micro)：{f1_score(y_dev, y_pred, average='micro'):.4f}")


# todo 5. 保存模型和向量化器

# 保存训练完成的随机森林模型
with open(conf.rf_model_save_path, "wb") as file:
    pickle.dump(model, file)

# 保存训练阶段拟合完成的TF-IDF向量化器
with open(conf.tfidf_model_save_path, "wb") as file:
    pickle.dump(tfidf, file)

# 提示保存成功
print("模型和向量化器保存成功！")