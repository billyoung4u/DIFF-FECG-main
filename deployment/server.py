import sys
import os
import uvicorn
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import resample
from sklearn.preprocessing import scale

# --- 1. 环境路径设置 (确保能引用上一级目录的模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 引入项目原有的工具
from web_system.inference_core import InferenceEngine
import data_util as du  # 复用原有的滤波逻辑
from config import cfg

# --- 2. 初始化 API ---
app = FastAPI(title="DIFF-FECG AI Service", version="1.0")

print("正在初始化 AI 引擎...")
# 这里的 model_name 和 db 请根据你实际最好的模型修改
# 注意：生产环境建议显式指定 model_name，不要让它猜
engine = InferenceEngine(model_name='mkf2_improved', db='addb')
print("AI 引擎加载完成！")


# --- 3. 定义数据协议 (Java 发过来的格式) ---
class EcgRequest(BaseModel):
    patient_id: str
    fs: int  # 医院数据的原始采样率 (例如 500)
    data: list[float]  # 原始电压值列表


# --- 4. 核心适配器函数 (Adapter) ---
def preprocess_adapter(raw_data, input_fs, target_fs=200):
    """
    将医院原始数据转换为模型需要的格式：
    1. 重采样到 200Hz
    2. 带通滤波 (7.5-75Hz) + 陷波 (50Hz)
    3. Z-Score 归一化
    """
    signal = np.array(raw_data)

    # A. 重采样 (Resample)
    if input_fs != target_fs:
        # 计算新的点数： 原长 * (目标频率 / 原始频率)
        num_samples = int(len(signal) * target_fs / input_fs)
        signal = resample(signal, num_samples)

    # B. 滤波 (Filtering)
    # 调用 data_util.py 中的逻辑
    # 注意：butter_bandpass_filter 期望输入是 (Channel, Length)，所以要 reshape
    signal = signal.reshape(1, -1)

    # 带通滤波 7.5-75Hz (参考 config.py 配置)
    signal = du.butter_bandpass_filter(signal, 7.5, 75, target_fs, axis=1)
    # 工频陷波 50Hz
    signal = du.iirnotch(signal, target_fs)

    # C. 归一化 (Normalization)
    # data_util.py 里用了 scale (Z-Score)
    signal = scale(signal, axis=1)

    return signal.flatten()  # 变回一维数组


# --- 5. 定义接口 ---
@app.post("/predict")
def predict(request: EcgRequest):
    try:
        # 1. 适配预处理
        clean_input = preprocess_adapter(request.data, request.fs)

        # 2. 长度校验与切片 (模型需要 1000 点)
        # 如果长度不足 1000，进行填充；如果超过，取最后 1000 (或根据业务逻辑调整)
        target_len = 1000
        if len(clean_input) < target_len:
            # 填充 0
            clean_input = np.pad(clean_input, (target_len - len(clean_input), 0), 'constant')
        elif len(clean_input) > target_len:
            # 取最后 1000 点 (实时性)
            clean_input = clean_input[-target_len:]

        # 3. AI 推理
        rec_fecg, peaks = engine.process_window(clean_input)
        fhr, rr = engine.calculate_metrics(peaks)

        # 4. 返回结果
        return {
            "code": 200,
            "msg": "success",
            "patient_id": request.patient_id,
            "result": {
                "fecg_signal": rec_fecg.tolist(),  # 重建的胎儿信号
                "r_peaks": peaks.tolist(),  # R峰位置 (基于 200Hz)
                "fhr_bpm": float(fhr),  # 胎心率
                "rr_ms": float(rr)  # RR间隔
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 500, "msg": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)