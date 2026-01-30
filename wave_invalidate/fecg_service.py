import torch
import numpy as np
from scipy import signal
import os
import sys
import math
import importlib.util

# --- 核心修复：将父目录加入路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# -------------------------------

# 引入配置
from config import cfg


# ==========================================
# 动态加载 runner
# ==========================================
def dynamic_import(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 拼凑 GetTrainTest-fecg.py 的绝对路径
script_path = os.path.join(parent_dir, "GetTrainTest-fecg.py")
runner = dynamic_import("GetTrainTest-fecg", script_path)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner.DEVICE = DEVICE


class FecgInferenceService:
    def __init__(self, model_name="addb_mkf2_improved_0_fecg_diff_0.5"):
        print("🔄 初始化推理服务...")
        self.model = self._load_model(model_name)
        self.params = runner.inference_schedule(self.model)
        print("✅ DIFF-FECG 服务已就绪")

    def _load_model(self, model_name):
        possible_dirs = [os.path.join("results", "model"), os.path.join("resource", "model"), "model"]
        model_dir = next((d for d in possible_dirs if os.path.exists(d)), None)

        if not model_dir:
            raise FileNotFoundError("❌ 找不到 results/model 或 resource/model 文件夹")

        if not model_name.endswith(".pt"):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
            if pt_files:
                target = next((f for f in pt_files if model_name in f), pt_files[0])
                model_name = target.replace(".pt", "")

        print(f"   -> 加载权重: {model_name}")
        model = runner.load_model(model_dir=model_dir, model_file=model_name)
        model = model.to(DEVICE)
        model.eval()
        return model

    def process_single_channel(self, raw_signal_full):
        """
        全量处理单通道信号，包含严格的 4 步后处理
        """
        # 1. 准备数据: 250Hz -> 1000Hz
        len_raw = len(raw_signal_full)
        len_model = len_raw * 4
        raw_1k = signal.resample(raw_signal_full, len_model)

        # 2. 切片推理 (Chunking Inference)
        CHUNK_SIZE = 2000
        full_fecg_1k = []
        num_chunks = math.ceil(len_model / CHUNK_SIZE)

        with torch.no_grad():
            alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar = self.params

            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min((i + 1) * CHUNK_SIZE, len_model)
                seg = raw_1k[start:end]
                current_seg_len = len(seg)

                if current_seg_len < CHUNK_SIZE:
                    pad_len = CHUNK_SIZE - current_seg_len
                    seg = np.pad(seg, (0, pad_len), 'constant')

                # 预处理 (去漂移+归一化，用于模型输入)
                seg_detrend = signal.detrend(seg)
                mean = np.mean(seg_detrend)
                std = np.std(seg_detrend) + 1e-6
                seg_norm = (seg_detrend - mean) / std

                seg_input = np.tile(seg_norm, (4, 1))
                seg_tensor = torch.from_numpy(seg_input).float().to(DEVICE)

                output = runner.predict(self.model, seg_tensor,
                                        alpha, beta, alpha_cum, sigmas, T,
                                        c1, c2, c3, delta, delta_bar,
                                        device=DEVICE)

                out_numpy = output[0, :].cpu().numpy()

                if current_seg_len < CHUNK_SIZE:
                    out_numpy = out_numpy[:current_seg_len]
                full_fecg_1k.append(out_numpy)

        # 拼接结果 (1000Hz)
        full_fecg_1k = np.concatenate(full_fecg_1k)

        # =========================================================
        # 🔥 核心修改：严格执行 4 步后处理逻辑
        # =========================================================

        # [步骤 1] 剔除 >100k 的坏点 (虽然模型输出通常较小，但这能防止意外爆炸)
        full_fecg_1k = np.clip(full_fecg_1k, -100000, 100000)

        # [步骤 2] 组合滤波 (在 1000Hz 下进行以获得更好效果)
        # A. 50Hz 陷波 (Q=30)
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=1000)
        full_fecg_1k = signal.filtfilt(b_notch, a_notch, full_fecg_1k)

        # B. 5-50Hz 带通滤波
        # 注意：这里使用 sosfiltfilt 保证零相位偏移
        sos_bp = signal.butter(4, [5, 50], btype='bandpass', fs=1000, output='sos')
        full_fecg_1k = signal.sosfiltfilt(sos_bp, full_fecg_1k)

        # 降采样回 250Hz (必须在滤波后进行，防止混叠)
        full_fecg_250 = signal.resample(full_fecg_1k, len_raw)

        # [步骤 3] 减去全局均值 (确保波形居中)
        full_fecg_250 = full_fecg_250 - np.mean(full_fecg_250)

        # [步骤 4] 忽略前1%和后1%的极值进行缩放 (标准化)
        # 这一步将输出波形归一化，使其幅度在一个标准范围内，
        # 方便 app.py 随后根据原始信号的幅度进行拉伸。
        p1, p99 = np.percentile(full_fecg_250, [1, 99])
        robust_amp = (p99 - p1) / 2.0

        if robust_amp > 1e-6:
            full_fecg_250 = full_fecg_250 / robust_amp

        return full_fecg_250