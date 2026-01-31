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

# 动态加载根目录下的 GetTrainTest-fecg.py
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
        Input: input_signal (前端处理后的单通道数据)
        Output: (fecg_display, peaks)
        """
        # 1. 域适应滤波 (1-75Hz)
        sos_model = signal.butter(4, [1.0, 75.0], btype='bandpass', fs=fs, output='sos')
        raw_for_model = signal.sosfiltfilt(sos_model, input_signal)

        # 2. 升采样 -> 1000Hz
        target_len = int(len(input_signal) * (self.fs_model / fs))
        raw_1k = signal.resample(raw_for_model, target_len)

        # 3. Z-Score 归一化
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

        # 5. 后处理
        sos_post = signal.butter(4, [5.0, 70.0], btype='bandpass', fs=1000, output='sos')
        fecg_clean = signal.sosfiltfilt(sos_post, fecg_1k)

        # 降采样回原始频率
        fecg_final = signal.resample(fecg_clean, len(input_signal))

        # 去直流
        fecg_final = fecg_final - np.mean(fecg_final)

        # [核心修正] 强制幅度缩放 -> 适配 +/- 5 的范围
        # 统计 99.5% 分位点，将其映射到 4.0 左右（留 1.0 的余量）
        abs_val = np.abs(fecg_final)
        p99 = np.percentile(abs_val, 99.5)

        if p99 < 1e-6: p99 = 1.0

        scale_factor = 4.0 / p99  # 改为 4.0 (适配 +/- 5 范围)
        fecg_display = fecg_final * scale_factor

        # [核心修正] 峰值检测阈值调整
        # 信号最大值约在 4.0 ~ 5.0
        # 绝对阈值设为 1.5 比较稳健
        min_dist = int(fs * 0.30)
        height_thresh = 1.5  # 改为 1.5

        peaks, _ = signal.find_peaks(fecg_display, distance=min_dist, height=height_thresh)

        return fecg_display, peaks