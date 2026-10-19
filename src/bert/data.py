import sys  # 用于注入项目根目录，支持直接运行本文件自检
from pathlib import Path  # 跨平台路径处理
import torch  # PyTorch 张量与设备管理
from torch.utils.data import Dataset, DataLoader  # 数据集基类与批量加载器

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 必须在 src 导入之前注入项目根目录

from src.config import Config  # 全项目统一路径与超参数配置（注意：config.py 导出的是类，不是实例）

config = Config()  # 实例化全局配置


# 1. 加载类别名称：class.txt 一行一个类别，行号即标签编号
def load_class_list():
    with open(config.class_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


# 2. 加载新闻样本：格式为「文本\t标签」，rsplit 只切最后一个制表符，防止正文含 \t 时切错
def load_samples(file_path):
    samples = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:  # 空行直接跳过
                continue
            parts = line.rsplit("\t", 1)  # 1 表示只切最后一个制表符，即从右边开始切
            if len(parts) != 2:  # 数据格式不符合要求时主动报错，带行号方便定位
                raise ValueError(f"第 {line_no} 行格式错误：{line}")
            text, label = parts
            try:
                label = int(label)  # 标签转 int，失败时报出行号
            except ValueError:
                raise ValueError(f"第 {line_no} 行标签不是整数：{line}")
            samples.append((text, label))  # 把 (原始文本, int标签) 组成元组追加到 samples 列表，供 Dataset/Tokenizer 后续使用
    return samples


# 3. 只返回原始文本和标签，Tokenizer 放到 collate_fn 中统一批量处理，避免逐条编码拖慢训练
class NewsDataset(Dataset):
    """自定义 Dataset：告诉 PyTorch 有多少数据、怎么取第 N 条"""

    def __init__(self, samples):  # 保存全部样本
        self.samples = samples

    def __len__(self):  # 样本总数
        return len(self.samples)

    def __getitem__(self, index):  # 取出第 index 条 (text, label)
        return self.samples[index]


# 4. 构建批处理函数：一个 Batch 批量 Tokenize，Batch 内按最长样本动态 Padding（而非固定 max_length）
def build_collate_fn(tokenizer):
    def collate_fn(batch):  # 这里的 batch 是一个列表，包含一个 Batch 的样本
        texts = [item[0] for item in batch]  # 提取一批文本
        labels = [item[1] for item in batch]  # 提取一批标签
        encoded = tokenizer(
            texts,
            padding=True,  # Batch 内动态 Padding
            truncation=True,  # 超长文本必须截断
            max_length=config.max_length,  # 截断上限
            return_tensors="pt",  # 直接返回 PyTorch Tensor
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)  # 标签转为 LongTensor 并入 Batch
        return encoded

    return collate_fn


# 5. 构建 DataLoader：加载样本 → Dataset → 批量迭代器
def build_dataloader(file_path, tokenizer, shuffle):
    dataset = NewsDataset(load_samples(file_path))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,  # 每批样本数
        shuffle=shuffle,  # 训练集打乱、验证集不打乱
        collate_fn=build_collate_fn(tokenizer),  # 批处理时统一 Tokenize
        num_workers=config.num_workers,  # 设置工作进程数，设置为 0 表示在主线程中进行数据加载
        pin_memory=str(config.device).startswith("cuda"),  # GPU 训练时锁页内存加速搬运；用 str 判断兼容 device 兜底为字符串的情况
    )
    return loader

if __name__ == "__main__":
    # 自检：验证数据加载链路
    class_list = load_class_list()
    print(f"类别数：{len(class_list)}")

    samples = load_samples(config.train_path)
    print(f"训练集样本数：{len(samples)}")
    print(samples[:5])

    dataset = NewsDataset(samples)
    print(f"Dataset 长度：{len(dataset)}，dataset[0] 可用：{dataset[0][1] == samples[0][1]}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
    loader = build_dataloader(config.train_path, tokenizer, shuffle=True)
    print(loader)
    print("[OK] data.py 自检通过")

