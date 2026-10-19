# 导入 sklearn 分类评估指标
from sklearn.metrics import (
    accuracy_score,  # 准确率
    precision_score,  # 精确率
    recall_score,  # 召回率
    f1_score,  # F1 值
    classification_report,  # 逐类详细分类报告
)


def calculate_metrics(y_true, y_pred, class_list):
    """计算 Macro 口径整体指标 + 逐类分类报告，返回字典供训练/评估脚本使用。"""
    labels = list(range(len(class_list)))  # 锁定完整类别集合：部分类别未出现在预测结果时，防止 target_names 数量不匹配报错

    accuracy = accuracy_score(y_true, y_pred)  # 准确率
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)  # Macro 精确率，除以 0 时结果记 0
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)  # Macro 召回率
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)  # Macro F1

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,  # 强制输出全部 10 个类别，避免个别类别缺失时崩溃
        target_names=class_list,  # 用类别名替代数字编号
        digits=4,  # 保留 4 位小数
        zero_division=0,  # 无预测样本的类别指标记 0 并跳过警告
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }


if __name__ == "__main__":
    # 自检：构造小规模真实/预测标签，验证指标计算与缺失类别场景不报错
    class_list = ["finance", "realty", "stocks", "education", "science",
                  "society", "politics", "sports", "game", "entertainment"]

    y_true = [0, 1, 2, 3, 4, 0, 1, 2]
    y_pred = [0, 1, 2, 3, 4, 0, 2, 2]  # 类别 5~9 完全未出现，模拟小样本评估

    metrics = calculate_metrics(y_true, y_pred, class_list)
    print(f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
          f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    print(metrics["report"])
    print("[OK] metrics.py 自检通过")
