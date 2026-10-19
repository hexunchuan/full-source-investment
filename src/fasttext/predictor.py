"""定义FastTextPredictor类，启动时加载一次模型，对单条文本做与训练一致的预处理并返回预测标签和置信概率。用于Http API接口"""

import sys  # 导入sys与Path，用于注入项目根目录
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 将项目根目录加入模块搜索路径（必须在src导入之前）

import fasttext  # 导入FastText
from src.fasttext.preprocess import preprocess_text  # 导入统一预处理函数
from src.config import Config  # 导入配置

conf = Config()  # 创建配置对象


class FastTextPredictor:

    def __init__(self, model_path, mode="word"):
        # 保存文本处理模式
        self.mode = mode

        # 程序启动时只加载一次模型，避免每预测一条数据就重新 load_model()
        self.model = fasttext.load_model(str(model_path))

    def predict(self, text):
        # 1. 使用训练阶段相同的预处理
        processed_text = preprocess_text(text, mode=self.mode)

        # 2. 模型预测，取 top-1 标签和概率
        labels, probabilities = self.model.predict(processed_text, k=1)

        # 3. 去掉 __label__ 前缀，获取标签（predict 返回的是按置信度排序的列表，k=1 时取第 [0] 个即最高置信度标签）
        label = labels[0].removeprefix("__label__")

        # 4. 获取置信概率（与 labels[0] 一一对应的概率，同为列表需取 [0] 并用 float 转换）
        probability = float(probabilities[0])

        # 5. 封装结果并返回
        return {"label": label, "probability": probability}


if __name__ == "__main__":
    # 固定使用词级 AutoTune 模型
    model_path = conf.model_dir / "fasttext_word_autotune.bin"

    # 创建预测器（模型只加载一次，词级模型用 word 模式）
    predictor = FastTextPredictor(model_path, mode="word")
    text = "内马尔最终没能获得大力神杯，但是他仍然是世界足坛最优秀的球员之一！"

    result = predictor.predict(text)
    print(f"预测类别：{result['label']}，置信度：{round(result['probability'], 4)}\n")
