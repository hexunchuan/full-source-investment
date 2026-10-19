"""使用默认超参数在词级训练数据上训练FastText Baseline模型，保存为fasttext_news.bin供API和评估统一加载。"""

import sys  # 导入sys与Path，用于注入项目根目录
from pathlib import Path
import fasttext  # 导入FastText

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 将项目根目录加入模块搜索路径

from src.config import Config  # 导入配置

config = Config()  # 创建配置对象


# 训练字符级 Baseline（默认参数）
def train_char_baseline():
    # 字符级训练数据路径
    model = fasttext.train_supervised(input=str(config.processed_data_dir / "train_char.txt"))
    return model


# 训练词级 Baseline（默认参数）
def train_word_baseline():
    # 词级训练数据路径
    model = fasttext.train_supervised(input=str(config.processed_data_dir / "train_word.txt"))
    return model


# 定义模型保存函数
def save_model(model, model_name):
    # 构造模型保存路径（固定文件名，供API和评估统一加载）
    model_path = config.model_dir / f"{model_name}.bin"

    # 保存模型并打印保存位置
    model.save_model(str(model_path))
    print(f"{model_name}模型已保存：", model_path)


if __name__ == "__main__":
    # 先训练词级 Baseline
    word_model = train_word_baseline()
    save_model(word_model, "fasttext_word_baseline")

    # 再训练字符级 Baseline
    char_model = train_char_baseline()
    save_model(char_model, "fasttext_char_baseline")
