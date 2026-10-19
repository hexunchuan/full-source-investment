"""BERT Flask预测API服务，启动时加载一次微调最佳模型，对外提供单条与批量预测接口。"""

import sys  # 用于注入项目根目录
import time  # 统计预测耗时
from pathlib import Path  # 跨平台路径处理

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 必须在 src 导入之前注入项目根目录

from flask import Flask, jsonify, request  # Flask相关 , request: 接收数据 , jsonify: 返回JSON数据

from src.bert.predictor import BertNewsPredictor  # 导入BERT预测器（封装模型+Tokenizer+元数据）

app = Flask(__name__)  # 创建Flask应用

predictor = BertNewsPredictor()  # 服务启动时只加载一次模型（首次加载需数秒）


@app.get("/health")  # 健康检查接口
def health():
    return jsonify({"status": "ok", "model": "bert-base-chinese-finetuned"})


@app.post("/predict")  # 单条新闻预测接口
def predict():
    data = request.get_json(silent=True)  # 获取JSON请求体，非法JSON返回None
    if not data:
        return jsonify({"code": 400, "message": "请求体必须为 JSON"}), 400

    text = data.get("text", "")  # 获取文本
    if not isinstance(text, str) or not text.strip():
        return jsonify({"code": 400, "message": "text 必须是非空字符串"}), 400

    start_time = time.perf_counter()  # 开始计时
    result = predictor.predict_one(text)  # 模型推理
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)  # 推理耗时（毫秒）

    # 返回统一结构（与FastText字段一致，便于三模型同口径对比）
    return jsonify({"code": 0, "message": "success",
                    "data": {"label": result["pred_class"], "probability": result["score"],
                             "elapsed_ms": elapsed_ms}})


@app.post("/predict_batch")  # 批量新闻预测接口
def predict_batch():
    data = request.get_json(silent=True)
    texts = data.get("texts") if data else None  # 获取文本数组
    if not isinstance(texts, list) or not texts:
        return jsonify({"code": 400, "message": "texts 必须是非空列表"}), 400

    start_time = time.perf_counter()  # 开始计时
    results = predictor.predict_batch(texts)  # 批量推理（一次前向，远快于循环单条）
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)  # 批量总耗时（毫秒）

    return jsonify({"code": 0, "message": "success",
                    "data": {"count": len(results), "elapsed_ms": elapsed_ms,
                             "avg_ms": round(elapsed_ms / len(results), 2),
                             "items": [{"text": r["text"], "label": r["pred_class"],
                                        "probability": r["score"]} for r in results]}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8005, debug=False)  # BERT专用端口8005（8001=RF，8003=FastText）
