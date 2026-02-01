import torch
import numpy as np
from scipy import signal
import os
import sys
import importlib.util

# ==========================================
# 1. 路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)

# 动态加载 GetTrainTest-fecg.py
runner_path = os.path.join(root_dir, "GetTrainTest-fecg.py")
if not os.path.exists(runner_path):
    runner_path = os.path.join(current_dir, "GetTrainTest-fecg.py")
    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"Critical: '{runner_path}' not found.")

spec = importlib.util.spec_from_file_location("GetTrainTest_fecg", runner_path)
runner = importlib.util.module_from_spec(spec)
sys.modules["GetTrainTest_fecg"] = runner
spec.loader.exec_module(runner)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner.DEVICE = DEVICE


class InferenceCore:
    def __init__(self, model_name=None):
        print(f"🔄 InferenceCore Initializing on {DEVICE}...")
        self.model = self._load_model(model_name)
        self.params = runner.inference_schedule(self.model)
        self.fs_model = 1000
        print("✅ InferenceCore Ready.")

    def _load_model(self, model_name):
        possible_dirs = [
            os.path.join(root_dir, "results", "model"),
            os.path.join(root_dir, "resource", "model"),
            os.path.join(root_dir, "model"),
            os.path.join(current_dir, "model")
        ]
        model_dir = next((d for d in possible_dirs if os.path.exists(d)), None)
        if not model_dir:
            raise FileNotFoundError("Model directory not found.")

        if not model_name:
            pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in {model_dir}")
            model_name = pt_files[0].replace(".pt", "")
            print(f"🔹 Auto-loading model: {model_name}")

        model = runner.load_model(model_dir=model_dir, model_file=model_name)
        model = model.to(DEVICE)
        model.eval()
        return model

    def predict_from_signal(self, input_signal, fs=250):
        """
        Input: input_signal (单位 uV)
        Output: (fecg_real_uv, peaks)
        """
        # 1. 宽带滤波
        sos_model = signal.butter(4, [1.0, 75.0], btype='bandpass', fs=fs, output='sos')
        raw_for_model = signal.sosfiltfilt(sos_model, input_signal)

        # 2. 升采样 -> 1000Hz
        target_len = int(len(input_signal) * (self.fs_model / fs))
        raw_1k = signal.resample(raw_for_model, target_len)

        # 3. Z-Score 归一化 (记录 sigma)
        mu = np.mean(raw_1k)
        sigma = np.std(raw_1k)
        if sigma < 1e-6: sigma = 1.0
        raw_1k_norm = (raw_1k - mu) / sigma

        # 4. 推理
        inp = np.tile(raw_1k_norm, (4, 1))
        inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar = self.params
            output = runner.predict(self.model, inp_tensor.squeeze(0),
                                    alpha, beta, alpha_cum, sigmas, T,
                                    c1, c2, c3, delta, delta_bar,
                                    device=DEVICE)

        fecg_1k = output[0, :].cpu().numpy()

        # 5. 后处理 (7.5-75Hz)
        sos_post = signal.butter(4, [7.5, 75.0], btype='bandpass', fs=1000, output='sos')
        fecg_clean = signal.sosfiltfilt(sos_post, fecg_1k)

        # 6. 降采样回 200Hz
        # 计算 200Hz 下的长度
        duration_sec = len(input_signal) / fs
        target_len_200 = int(duration_sec * 200)
        fecg_final = signal.resample(fecg_clean, target_len_200)

        # 去直流
        fecg_final = fecg_final - np.mean(fecg_final)

        # 7. 物理还原
        # fecg_final 是归一化的形状，sigma 是原信号的 uV 能量
        fecg_real_uv = fecg_final * sigma

        # =======================================================
        # [关键修复] 峰值检测 V2.0
        # =======================================================
        # 1. 最小距离: 0.25s (对应 240bpm，防止漏掉快心率)
        min_dist = int(200 * 0.25)

        # 2. 计算信号的动态范围 (Peak-to-Peak)
        # 用 PTP 而不是 Max，因为基线可能不完全在0
        dynamic_range = np.ptp(fecg_real_uv)

        if dynamic_range > 0.5:  # 只有当信号波动大于 0.5uV 才检测
            # 策略：突起度 (Prominence)
            # 只要波峰比它相邻的低谷高出 Range 的 25%，就认为是 R 峰
            # 这比绝对高度阈值更靠谱
            prominence_val = dynamic_range * 0.25

            # 同时保留一个很低的绝对高度门槛，过滤掉 0 附近的微小噪声
            height_val = np.max(fecg_real_uv) * 0.15

            peaks, _ = signal.find_peaks(fecg_real_uv, distance=min_dist, prominence=prominence_val, height=height_val)
        else:
            peaks = np.array([])

        print(f"DEBUG: Signal PTP={dynamic_range:.2f}uV, Detected Peaks={len(peaks)}")

        return fecg_real_uv, peaks