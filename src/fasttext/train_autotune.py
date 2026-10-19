"""使用FastText AutoTune在词级数据上自动调参训练（指定验证集和调参时长），保存为fasttext_word_autotune.bin。"""

import fasttext                             # 导入FastText
import sys                                  # 导入sys与Path，用于注入项目根目录
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 将项目根目录加入模块搜索路径

from src.config import Config               # 导入配置

config = Config()                           # 创建配置对象


# 定义 AutoTune 训练函数
def train_autotune(mode="word", duration=300):
    # 训练集与验证集路径
    train_path = config.processed_data_dir / f"train_{mode}.txt"
    dev_path = config.processed_data_dir / f"dev_{mode}.txt"

    # 开始自动调参
    model = fasttext.train_supervised(
        input=str(train_path),                   # 训练集   str的作用是将Path对象转换为字符串
        autotuneValidationFile=str(dev_path),    # 验证集   str的作用是将Path对象转换为字符串
        autotuneDuration=duration,               # 自动调参持续时间（秒）
        verbose=3,                               # 打印详细日志 3代表详细
    )
    return model


if __name__ == "__main__":
    # 依次训练词级和字级 AutoTune 模型
    for level in ("word", "char"):
        # 训练当前级别的 AutoTune 模型
        model = train_autotune(mode=level, duration=300)

        # 保存模型并输出保存位置
        model_path = config.model_dir / f"fasttext_{level}_autotune.bin"
        model.save_model(str(model_path))
        print("模型保存完成：", model_path)
