"""基于Flask启动RF预测API服务（端口8001），接收JSON文本并调用predict_fun返回预测类别。"""

from flask import Flask, request, jsonify                          # Flask相关 , request: 接收数据 , jsonify: 返回JSON数据
import pandas as pd                                                 # 表格数据
import sys                                                          # 用于注入项目根目录
import warnings                                                     # 忽略警告
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # 将项目根目录加入模块搜索路径

from src.rf.predictor import predict_fun                            # 导入预测函数

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    # 获取 JSON 输入
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field in JSON'}), 400

    if not str(data['text']).strip():
        return jsonify({'error': 'text field is empty'}), 400

    # 调用预测函数（异常时返回 500 和 JSON 错误信息，便于前端展示）
    try:
        result = predict_fun(data)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # 返回 JSON 结果
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8001, debug=True)
