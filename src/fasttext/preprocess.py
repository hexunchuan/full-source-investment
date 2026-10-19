"""将原始TXT数据转换为FastText训练格式，支持char（字符级）和word（jieba词级）两种模式，生成train/dev/test的__label__格式TXT文件。"""

import sys  # 导入sys与Path，用于注入项目根目录
from pathlib import Path
import jieba  # 用来完成中文分词

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 保证任意工作目录下都能 import src.config

from src.config import Config  # 导入配置类

config = Config()  # 创建配置对象


# 加载类别映射
def load_label_mapping(class_path):
    # 创建空字典，用来保存数字ID -> 类别名
    id_to_name = {}

    # 打开类别文件，逐行读取类别名
    with open(class_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            # 去掉换行符后保存：数字ID -> 类别名
            id_to_name[idx] = line.strip()

    # 返回类别映射 {0:"finance",1:"realty"...}
    return id_to_name


# 定义统一文本预处理函数，支持 char（字符级）和 word（jieba词级）两种模式
def preprocess_text(text, mode="word"):
    # 去掉首尾空白
    text = text.strip()

    # 字符级：一句话拆成一个个字符；词级：使用 jieba 分词
    if mode == "char":
        tokens = list(text)  # 拆成一个个字符
    elif mode == "word":
        tokens = jieba.lcut(text)  # 使用 jieba 分词
    else:
        # 参数不正确时主动报错
        raise ValueError("mode 只能是 'char' 或 'word'")

    # 去掉空白 token，并使用空格连接
    processed_text = " ".join(token for token in tokens if token.strip())

    # 返回处理结果
    return processed_text


# 构建 FastText 数据文件
def build_fasttext_file(input_path, output_path, id_to_name, mode="word"):
    # 统计样本数量
    count = 0

    # 同时打开输入文件和输出文件
    with open(input_path, "r", encoding="utf-8") as input_file, \
            open(output_path, "w", encoding="utf-8") as output_file:

        # 逐行读取数据
        for line in input_file:
            # 去掉首尾空白，跳过空行
            line = line.strip()
            if not line:
                continue

            # 原始格式：文本\t数字标签  例如：基金高管离职带来业绩地震   0
            text, label = line.split("\t")

            # 数字标签转 int，并找到对应的英文类别名
            label_name = id_to_name[int(label)]

            # 构造 FastText 一行数据：__label__类别名 预处理后的文本  例如：__label__education 送考 队伍 成海 高考 场外 那些 煎熬 的 心
            fasttext_line = f"__label__{label_name} {preprocess_text(text, mode=mode)}"

            # 写入文件，数量 +1 注意：每行数据末尾会自动添加换行符
            output_file.write(fasttext_line + "\n")
            count += 1

    # 输出处理结果
    print(f"处理完成：{input_path}")
    print(f"保存位置：{output_path}")
    print(f"样本数量：{count}")


# 批量处理 train / dev / test
def build_all_data(mode="word"):
    # 加载类别映射，确定输出文件后缀
    id_to_name = load_label_mapping(config.class_path)
    suffix = "char" if mode == "char" else "word"

    # 依次处理训练集、验证集、测试集
    build_fasttext_file(config.train_path, config.processed_data_dir / f"train_{suffix}.txt", id_to_name, mode)
    build_fasttext_file(config.dev_path, config.processed_data_dir / f"dev_{suffix}.txt", id_to_name, mode)
    build_fasttext_file(config.test_path, config.processed_data_dir / f"test_{suffix}.txt", id_to_name, mode)


if __name__ == "__main__":
    # 生成字符级数据
    build_all_data(mode="char")

    # 生成词级数据
    build_all_data(mode="word")
