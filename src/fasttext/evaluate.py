"""依次评估词级/字级的 Baseline 与 AutoTune 模型，逐条预测计算准确率并打印分类报告。"""

import sys  # 导入sys与Path，用于注入项目根目录
from pathlib import Path
import fasttext  # 导入FastText
from sklearn.metrics import accuracy_score, classification_report  # 准确率与分类评估报告

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 将项目根目录加入模块搜索路径

from src.config import Config  # 导入配置

config = Config()  # 创建配置对象


# 定义模型评估函数
def evaluate_model(model_path, test_path):
    # 加载模型
    model = fasttext.load_model(str(model_path))

    # ===== FastText 原生评估（C++层直接读取测试文件，返回 N、P@1、R@1）=====
    n_samples, precision_at_1, recall_at_1 = model.test(str(test_path))
    print(f"[FastText原生] N={n_samples}, P@1={round(precision_at_1, 4)}, R@1={round(recall_at_1, 4)}")

    # 保存真实标签与预测标签
    y_true = []
    y_pred = []

    # 一行一行读取测试集
    with open(test_path, "r", encoding="utf-8") as file:
        # 逐行处理
        for line in file:
            # 去掉行首行尾的空白字符
            line = line.strip()
            # 跳过空行
            if not line:
                continue

            # 一行格式：__label__sports 文本，只切第一个空格
            label, text = line.split(" ", 1)

            # 模型预测，保存真实标签和预测标签 ，pred_labels表示预测标签，pred_probs表示预测概率，k=1表示只保存一个预测结果
            pred_labels, pred_probs = model.predict(text, k=1)
            y_true.append(label)
            y_pred.append(pred_labels[0])

    # 计算并打印准确率，打印完整分类报告
    accuracy = accuracy_score(y_true, y_pred)
    print("[sklearn] Accuracy：", round(accuracy, 4))
    print(classification_report(y_true, y_pred))
    return accuracy


if __name__ == "__main__":
    # 待评估模型清单：(名称, 模型文件, 匹配的测试集级别)
    models_to_eval = [
        ("词级 Baseline", config.model_dir / "fasttext_word_baseline.bin", "word"),
        ("字级 Baseline", config.model_dir / "fasttext_char_baseline.bin", "char"),
        ("词级 AutoTune", config.model_dir / "fasttext_word_autotune.bin", "word"),
        ("字级 AutoTune", config.model_dir / "fasttext_char_autotune.bin", "char"),
    ]

    # 逐个模型评估，词/字分别匹配对应级别的测试集
    for name, model_path, level in models_to_eval:
        print("=" * 50)
        print(f"评估模型：{name}（{model_path.name}）")
        print("=" * 50)

        # 模型文件不存在则跳过，避免中断整体评估
        if not Path(model_path).is_file():
            print("模型文件缺失，跳过，请先运行 train_baseline.py / train_autotune.py\n")
            continue

        # 测试集与模型级别保持一致
        test_path = config.processed_data_dir / f"test_{level}.txt"
        evaluate_model(model_path, test_path)
        print()
