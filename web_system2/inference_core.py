import torch
import numpy as np
from scipy import signal
import os
import sys
import importlib.util

# --- 路径修复 (确保能找到上级目录的 config 和 GetTrainTest-fecg) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import cfg


# 动态加载 runner
def dynamic_import(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = dynamic_import("GetTrainTest-fecg", os.path.join(parent_dir, "GetTrainTest-fecg.py"))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner.DEVICE = DEVICE


class InferenceCore:
    def __init__(self, model_name="addb_mkf2_improved_0_fecg_diff_0.5"):
        print("🔄 初始化推理核心...")
        self.model = self._load_model(model_name)
        self.params = runner.inference_schedule(self.model)
        self.fs_model = 1000  # 模型需要的采样率
        self.fs_raw = 250  # 原始 TXT 数据的采样率
        print("✅ 推理核心就绪")

    def _load_model(self, model_name):
        # 寻找模型路径
        possible_dirs = [
            os.path.join(parent_dir, "results", "model"),
            os.path.join(parent_dir, "resource", "model"),
            os.path.join(parent_dir, "model")
        ]
        model_dir = next((d for d in possible_dirs if os.path.exists(d)), None)
        if not model_dir: raise FileNotFoundError("Model directory not found")

        if not model_name.endswith(".pt"):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
            if pt_files: model_name = pt_files[0].replace(".pt", "")

        model = runner.load_model(model_dir=model_dir, model_file=model_name)
        model = model.to(DEVICE)
        model.eval()
        return model

    def strict_preprocessing(self, data):
        """
        [严格预处理流程] (仅用于母体心电的清洗与网页显示)
        1. 剔除 > 100k 的坏点
        2. 带通 5-50Hz + 陷波 50/60Hz
        """
        # 1. 坏点剔除
        data = np.clip(data, -100000, 100000)

        # 2. 滤波 (在 250Hz 下进行)
        # 50Hz 陷波
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=self.fs_raw)
        data = signal.filtfilt(b_notch, a_notch, data)
        # 60Hz 陷波
        b_notch2, a_notch2 = signal.iirnotch(w0=60.0, Q=30.0, fs=self.fs_raw)
        data = signal.filtfilt(b_notch2, a_notch2, data)

        # 5-50Hz 带通
        sos = signal.butter(4, [5, 50], btype='bandpass', fs=self.fs_raw, output='sos')
        data = signal.sosfiltfilt(sos, data)

        return data

    def process_segment(self, raw_segment):
        """
        处理一个时间窗口的数据
        Input: raw_segment (numpy array, 250Hz)
        Output: raw_clean (250Hz), fecg_processed (200Hz)
        """
        # ==========================================
        # 1. 准备网页显示的母体心电 (Clean Data)
        # ==========================================
        # 这里进行严格清洗，为了让医生看得清楚
        raw_clean = self.strict_preprocessing(raw_segment)
        raw_clean = raw_clean - np.mean(raw_clean)

        # 计算缩放因子 (基于干净的母体信号)
        p1, p99 = np.percentile(raw_clean, [1, 99])
        scale_factor = (p99 - p1) / 2.0
        if scale_factor < 1e-6: scale_factor = 1.0

        # ==========================================
        # 2. 准备 AI 模型输入 (Raw Data)
        # ==========================================
        # 🔥【核心修正】：这里必须使用 raw_segment (原始含噪数据)！
        # 如果喂给模型 raw_clean，模型会因为数据分布不匹配而失效，
        # 导致输出结果也是母体心电。

        len_raw = len(raw_segment)
        len_model = len_raw * 4

        # 使用原始数据重采样
        raw_1k = signal.resample(raw_segment, len_model)

        # 归一化 (Z-score) 是模型必须的
        # Detrend 一下防止极度漂移影响归一化，但不做强滤波
        raw_1k_detrend = signal.detrend(raw_1k)
        model_input_norm = (raw_1k_detrend - np.mean(raw_1k_detrend)) / (np.std(raw_1k_detrend) + 1e-6)

        # 构造 Tensor
        inp = np.tile(model_input_norm, (4, 1))
        inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(DEVICE)

        # ==========================================
        # 3. 执行推理
        # ==========================================
        with torch.no_grad():
            alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar = self.params
            output = runner.predict(self.model, inp_tensor.squeeze(0),
                                    alpha, beta, alpha_cum, sigmas, T,
                                    c1, c2, c3, delta, delta_bar,
                                    device=DEVICE)

        fecg_1k = output[0, :].cpu().numpy()

        # ==========================================
        # 4. 后处理 FECG (按您的新要求)
        # ==========================================

        # (1) 带通滤波 7.5Hz - 75Hz
        # 这一步能有效去除生成的低频伪影和极高频噪声
        sos_bp = signal.butter(4, [7.5, 75], btype='bandpass', fs=1000, output='sos')
        fecg_filtered = signal.sosfiltfilt(sos_bp, fecg_1k)

        # (2) 重采样到 200Hz
        # 目标点数 = 原始时间长度(秒) * 200Hz
        target_len = int((len_raw / 250.0) * 200)
        fecg_200 = signal.resample(fecg_filtered, target_len)

        # (3) 恢复幅度
        # 这一步是为了让 FECG 在网页上能以 uV 为单位显示
        fecg_final = (fecg_200 - np.mean(fecg_200)) * scale_factor

        # 返回：[清洗后的母体心电], [提取出的胎儿心电]
        return raw_clean, fecg_final