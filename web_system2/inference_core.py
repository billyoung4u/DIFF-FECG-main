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

    def strict_preprocessing(self, data, fs):
        """
        [通用预处理]
        让原始数据在频谱特性上尽可能接近 ADDB 数据集
        """
        # 1. 极值截断 (去除像脱落一样的巨大幅度突变)
        # 用分位数裁剪比固定数值更稳健
        p1, p99 = np.percentile(data, [0.5, 99.5])
        data = np.clip(data, p1, p99)

        # 2. 移除直流偏置 (去基线)
        # 方法：先减去均值，再用高通滤波
        data = data - np.mean(data)

        # 3. 组合滤波 (关键步骤)
        # ADDB 频带通常在 0.05 - 100Hz 之间。
        # 真实环境噪声大，建议：
        # - 高通 1.0Hz (去除顽固基线漂移)
        # - 低通 75Hz (去除肌电干扰，胎儿QRS能量主要在10-50Hz)
        # - 陷波 50Hz (去除电源干扰)

        # A. 50Hz 陷波 (根据你所在地的市电频率修改，国内50，国外部分60)
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
        data = signal.filtfilt(b_notch, a_notch, data)

        # B. 带通滤波 1Hz - 75Hz
        sos = signal.butter(4, [1.0, 75.0], btype='bandpass', fs=fs, output='sos')
        data = signal.sosfiltfilt(sos, data)

        return data

    def process_segment(self, raw_segment):
        """
        处理一个时间窗口的数据
        """
        # ==========================================
        # 1. 统一清洗 (让数据像 ADDB)
        # ==========================================
        # 先在 250Hz 下清洗，效果最好，计算量也小
        clean_segment = self.strict_preprocessing(raw_segment, fs=self.fs_raw)

        # ==========================================
        # 2. 准备网页显示的母体心电
        # ==========================================
        # 计算缩放因子用于还原显示
        p1, p99 = np.percentile(clean_segment, [1, 99])
        scale_factor = (p99 - p1) / 2.0
        if scale_factor < 1e-6: scale_factor = 1.0

        # ==========================================
        # 3. 准备 AI 模型输入 (升采样 + 归一化)
        # ==========================================
        len_raw = len(clean_segment)
        # 目标长度：因为模型是按 1000Hz 训练的，所以点数要 * 4
        target_len = int(len_raw * (self.fs_model / self.fs_raw))

        # A. 升采样 (250Hz -> 1000Hz)
        # 注意：使用 clean_segment 进行重采样，不要用 raw_segment
        raw_1k = signal.resample(clean_segment, target_len)

        # B. Z-Score 归一化 (Domain Adaptation 的核心)
        # 这一步强制让数据分布符合 N(0, 1)，消除幅度差异
        mu = np.mean(raw_1k)
        sigma = np.std(raw_1k)
        model_input_norm = (raw_1k - mu) / (sigma + 1e-6)

        # 构造 Tensor
        inp = np.tile(model_input_norm, (4, 1))  # 复制4份 (Batch=4)
        inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(DEVICE)

        # ==========================================
        # 4. 执行推理
        # ==========================================
        with torch.no_grad():
            alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar = self.params
            output = runner.predict(self.model, inp_tensor.squeeze(0),
                                    alpha, beta, alpha_cum, sigmas, T,
                                    c1, c2, c3, delta, delta_bar,
                                    device=DEVICE)

        fecg_1k = output[0, :].cpu().numpy()

        # ==========================================
        # 5. 后处理 FECG
        # ==========================================
        # 再次滤波清理生成结果
        sos_bp = signal.butter(4, [5.0, 70.0], btype='bandpass', fs=1000, output='sos')
        fecg_filtered = signal.sosfiltfilt(sos_bp, fecg_1k)

        # 降采样回 200Hz (为了显示或其他用途)
        final_len = int((len_raw / self.fs_raw) * 200)
        fecg_200 = signal.resample(fecg_filtered, final_len)

        # 恢复幅度 (可选，为了视觉上匹配输入)
        fecg_final = (fecg_200 - np.mean(fecg_200)) * (scale_factor * 0.5)  # 胎儿信号通常比母体弱

        return clean_segment, fecg_final

    def calculate_fhr_metrics(self, fecg_signal, fs=200):
        """
        [新增] 根据 FECG 信号计算心率指标
        :param fecg_signal: 200Hz 的胎儿心电信号 (numpy array)
        :param fs: 采样率，默认为 200Hz
        :return: 包含 'bpm' (心率) 和 'rr_mean' (平均RR间隔) 的字典
        """
        # 1. 寻找 R 峰
        # distance: 设置最小峰间距。胎儿心率较快 (110-180bpm)，
        # 180bpm = 3Hz = 0.33s。为了安全起见，设置最小间距为 0.25s (240bpm)
        min_distance = int(fs * 0.25)

        # height: 动态阈值，避免噪声干扰
        threshold = np.max(fecg_signal) * 0.4

        peaks, _ = signal.find_peaks(fecg_signal, distance=min_distance, height=threshold)

        if len(peaks) < 2:
            return None  # 峰值太少，无法计算

        # 2. 计算 RR 间隔 (单位：秒)
        rr_intervals = np.diff(peaks) / fs

        # 3. 计算指标
        mean_rr = np.mean(rr_intervals)
        if mean_rr == 0: return None

        bpm = 60.0 / mean_rr

        return {
            "bpm": bpm,  # 实时心率 (BPM)
            "rr_mean": mean_rr,  # 平均 RR 间隔 (秒)
            "rr_std": np.std(rr_intervals) * 1000  # RR 变异性 (ms)
        }