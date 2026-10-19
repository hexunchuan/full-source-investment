"""基于Flask启动FastText预测API服务（端口8003），提供/health健康检查和/predict新闻分类接口。"""

from flask import Flask, jsonify, request   # 导入Flask，jsonify：返回JSON，request：接收请求
import sys                                  # 用于注入项目根目录
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 将项目根目录加入模块搜索路径
from src.fasttext.predictor import FastTextPredictor         # 导入预测器
from src.config import Config                                # 导入统一路径配置

# 创建 Flask 应用
app = Flask(__name__)

# 程序启动时加载一次模型（使用词级 AutoTune 模型，路径由 Config 统一管理）
config = Config()
predictor = FastTextPredictor(
    model_path=config.fasttext_word_autotune_path,
    mode="word",
)


# 健康检查接口
@app.get("/health")
def health():
    # 返回服务状态
    return jsonify({"status": "ok"})


# 新闻分类接口
@app.post("/predict")
def predict():
    # 获取 JSON 请求体，判断是否存在
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"code": 400, "message": "请求体必须为 JSON"}), 400   # 显式指定HTTP状态码

    # 获取 text 并判断是否为空
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"code": 400, "message": "text 不能为空"}), 400   # 显式指定HTTP状态码

    # 模型预测并返回 JSON
    result = predictor.predict(text)
    return jsonify({"code": 0, "message": "success", "data": result})  # 隐式指定HTTP状态码200，成功


# 启动 Flask：host 允许其他机器访问，port 为服务端口
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003, debug=True)  # 启动 Flask 服务，允许其他机器访问，端口为 8003，开启调试模式
