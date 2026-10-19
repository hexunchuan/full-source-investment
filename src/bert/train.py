import sys  # 用于注入项目根目录
import json  # 保存训练元数据
import random  # Python 随机数
from pathlib import Path  # 跨平台路径处理

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 必须在 src 导入之前注入项目根目录

import numpy as np  # NumPy 随机数
import torch  # PyTorch 训练框架
from tqdm import tqdm  # 进度条
from transformers import (  # Hugging Face 分词器与预训练模型
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.config import Config  # 全项目统一路径与超参数配置
from src.bert.data import load_class_list, build_dataloader  # BERT 数据链路
from src.bert.metrics import calculate_metrics  # BERT 指标计算

config = Config()  # 实例化全局配置（config.py 只导出类，不导出实例）


# ============================================================
# 1. 固定随机种子：保证训练结果可复现
# ============================================================

def set_seed(seed):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # PyTorch CPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 固定所有 GPU 的随机种子


# ============================================================
# 2. 将一个 Batch 移动到设备
# ============================================================

def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}  # 对 Batch 中所有 Tensor 调用 to(device)


# ============================================================
# 3. 模型评估：遍历数据收集预测与真实标签，返回 Macro 指标 + 分类报告
# ============================================================

def evaluate(model, data_loader, device, class_list):
    model.eval()  # 切换到评估模式
    all_preds, all_labels = [], []
    with torch.inference_mode():  # 推理阶段不计算梯度
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = move_batch_to_device(batch, device)
            preds = torch.argmax(model(**batch).logits, dim=1)  # 最大 logits 对应预测类别
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    return calculate_metrics(all_labels, all_preds, class_list)


# ============================================================
# 4. 训练主流程
# ============================================================

def train():
    set_seed(config.seed)

    # ---- 类别列表与标签映射 ----
    class_list = load_class_list()
    num_classes = len(class_list)
    id2label = {index: name for index, name in enumerate(class_list)}  # ID -> 类别名称
    label2id = {name: index for index, name in enumerate(class_list)}  # 类别名称 -> ID

    # ---- 加载 Tokenizer 与预训练模型（+ 分类头） ----
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_path,  # 本地 BERT 路径
        num_labels=num_classes,  # 分类类别数量
        id2label=id2label,  # 保存标签映射，推理时可直接还原类别名
        label2id=label2id,
    )
    model.to(config.device)  # 模型移动到 GPU / CPU

    # ---- 构建 DataLoader：训练集必须打乱，验证集不打乱 ----
    train_loader = build_dataloader(config.train_path, tokenizer, shuffle=True)
    dev_loader = build_dataloader(config.dev_path, tokenizer, shuffle=False)

    # ---- 优化器与学习率调度（Warmup + 线性衰减） ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.num_epochs  # 总训练 Step 数
    # ---- 学习率调度器 ----
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),  # Warmup Step 数
        num_training_steps=total_steps,
    )

    # ---- 最佳验证集指标 ----
    # ---- Early Stopping 状态 ----
    best_dev_f1 = -1.0  # 最佳验证集 F1
    no_improve_epochs = 0  # 连续未提升轮数

    for epoch in range(config.num_epochs):
        model.train()  # 切换回训练模式
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch in progress_bar:
            batch = move_batch_to_device(batch, config.device)
            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            outputs = model(**batch)  # Batch 含 labels，模型自动计算 CrossEntropyLoss
            loss = outputs.loss
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()  # 参数更新
            scheduler.step()  # 学习率更新
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")  # 进度条实时展示 Loss

        avg_train_loss = total_loss / len(train_loader)  # 当前 Epoch 平均 Loss

        # ---- 每个 Epoch 结束后跑验证集 ----
        dev_metrics = evaluate(model, dev_loader, config.device, class_list)
        dev_f1 = dev_metrics["f1"]
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
              f"Dev Accuracy: {dev_metrics['accuracy']:.4f} | Dev Macro-F1: {dev_f1:.4f}")

        # ---- 判断是否刷新最佳成绩 ----
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            no_improve_epochs = 0  # 重置未提升次数

            model.save_pretrained(config.bert_best_model_dir)  # 保存最佳模型（id2label 已写入 config.json）
            tokenizer.save_pretrained(config.bert_best_model_dir)  # Tokenizer 必须一起保存

            metadata = {  # 保存项目元数据，供预测端还原类别与截断长度
                "class_list": class_list,
                "max_length": config.max_length,
                "best_dev_f1": best_dev_f1,
                "pretrained_model": config.pretrained_model_path,
            }
            with open(config.bert_best_model_dir / "metadata.json", "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)
            print("保存新的最佳模型！")
        else:
            no_improve_epochs += 1  # 连续未提升次数 +1
            print(f"验证集没有提升：{no_improve_epochs}/{config.patience}")
            if no_improve_epochs >= config.patience:  # 触发 Early Stopping，提前结束训练
                print("触发 Early Stopping，停止训练。")
                break


# ============================================================
# 5. 主程序
# ============================================================

if __name__ == "__main__":
    train()  # 开始训练
