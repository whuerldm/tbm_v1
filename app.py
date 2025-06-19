import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 GPU，避免显存爆炸

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from datetime import datetime

# 初始化 Flask 应用
app = Flask(__name__)
model = None  # 全局模型变量（惰性加载）
MODEL_VERSION = "1.0"
MODEL_LOADED_AT = None


def load_model_once():
    """确保模型只加载一次（热重启时不重复加载）"""
    global model, MODEL_LOADED_AT
    if model is None:
        model = tf.keras.models.load_model("../../../tbm_model_api/tbm_model.h5")
        MODEL_LOADED_AT = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@app.route("/", methods=["GET"])
def index():
    return "✅ TBM 围岩等级预测服务已启动，请使用 POST /predict 接口进行预测。"


@app.route("/health", methods=["GET"])
def health():
    """简单健康检查接口"""
    try:
        load_model_once()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/version", methods=["GET"])
def version():
    """模型版本信息"""
    return jsonify({
        "model_version": MODEL_VERSION,
        "loaded_at": MODEL_LOADED_AT or "未加载"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model_once()

        data = request.get_json(force=True)
        features = data.get("features")

        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": "请提供数值类型的 features 数组，例如: [12, 200, 0.5, 1]"}), 400

        input_data = np.array([features], dtype=np.float32)
        prediction = model.predict(input_data)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # 生产环境去掉 debug=True
