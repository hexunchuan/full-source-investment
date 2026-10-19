"""加载TF-IDF向量化器和随机森林模型，封装单条文本的分词+向量化+预测函数，供Flask API调用。"""

import sys                                                          # 导包
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # 将项目根目录加入模块搜索路径

import pandas as pd                                                 # 读取和处理数据的
import pickle                                                       # 用于模型和向量化器的序列化存储
import jieba                                                        # 中文分词
import warnings                                                     # 忽略警告
from src.config import Config                                       # 配置文件

warnings.filterwarnings('ignore')

# todo 1.设置pandas显示选项 和 加载配置文件.
# 1. 设置pandas显示所有列.
pd.set_option('display.max_columns', None)         # 确保所有列可见
# 2. 加载配置文件.
conf = Config()

# todo 2. 加载模型和向量化器.
# 1. 加载模型
with open(conf.rf_model_save_path, 'rb') as f:
    rf = pickle.load(f)

# 2. 加载向量化器(tf-idf)
with open(conf.tfidf_model_save_path, 'rb') as f:
    tfidf = pickle.load(f)

# 3. 加载标签索引和名称的对应关系, 即: {0: 'finance', 1:'realty'..., 9:'entertainment'}
#    模块级只加载一次, 避免每次预测请求都读取文件
with open(conf.class_data_path, encoding='utf-8') as f:
    id2class = {i: line.strip() for i, line in enumerate(f) if line.strip()}


# todo 3. 定义预测函数 -> 充当flask的视图函数.
def predict_fun(data):          # data: 就是待预测的文本数据., 例如: {'text': '体验2D巅峰 倚天屠龙记十大创新概览'}
    # 1. jieba分词
    words = ' '.join(jieba.lcut(data['text'])[:30])

    # 2. 对数据进行向量化.
    features = tfidf.transform([words])

    # 3. 预测.
    y_pred_id = rf.predict(features)[0]

    # 4. 获取预测分类的名字(类别名)
    y_pred = id2class[y_pred_id]

    # 5. 给原始数据新增1列(预测的类别名), 并返回.
    data['pred_class'] = y_pred
    return data


# todo 4. 测试代码.
if __name__ == '__main__':
    # 1. 定义变量, 记录: 待预测的文本数据.
    data = {'text': '体验2D巅峰 倚天屠龙记十大创新概览'}
    # 2. 调用预测函数.
    result = predict_fun(data)
    # 3. 输出结果.
    print(result)