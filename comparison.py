import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
import json
import pandas as pd
from scipy import signal

# 引入配置
from config import cfg


# ==========================================
# 1. 基础工具
# ==========================================
def dynamic_import(module_name, file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = dynamic_import("GetTrainTest-fecg", "GetTrainTest-fecg.py")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner.DEVICE = DEVICE


# ==========================================
# 2. 鲁棒数据处理器 (抗噪版)
# ==========================================
class RobustOpenBCIProcessor:
    def __init__(self, model_fs=1000):
        self.model_fs = model_fs

    def clean_signal(self, data):
        """
        Pro 版清洗：带通滤波 (0.5Hz - 100Hz)
        """
        # 1. 处理 NaN 和 极端值
        data = np.nan_to_num(data)
        data = np.clip(data, -180000, 180000)

        # 2. 定义带通滤波器
        sos = signal.butter(4, [0.5, 100], btype='bandpass', fs=250, output='sos')

        # 3. 执行滤波
        try:
            clean_data = signal.sosfiltfilt(sos, data)
        except Exception as e:
            print(f"      ⚠️ 滤波失败，回退到简单 Detrend: {e}")
            clean_data = signal.detrend(data)

        return clean_data

    def load_openbci_txt(self, filepath):
        try:
            print(f"   -> 解析 OpenBCI: {os.path.basename(filepath)}")
            # 读取
            df = pd.read_csv(filepath, comment='%', header=0, skipinitialspace=True)
            df.columns = df.columns.str.strip()  # 去除列名空格

            # 找 EXG 列
            exg_cols = [c for c in df.columns if 'EXG Channel' in c]
            if not exg_cols:
                # 盲取 1-8 列
                raw_data = df.iloc[:, 1:9].values.T
                col_names = [f"Ch{i}" for i in range(8)]
            else:
                raw_data = df[exg_cols].values.T
                col_names = exg_cols

            # --- 智能筛选通道 ---
            valid_indices = []
            print("   -> 通道质量检查:")
            for i in range(raw_data.shape[0]):
                ch_data = raw_data[i, :]
                mean_val = np.mean(ch_data)

                # 判据：如果均值接近 +/- 187500，说明彻底断连 (Dead)
                if abs(mean_val) > 150000:
                    print(f"      ❌ {col_names[i]}: 坏死 (Mean={mean_val:.0f}) -> 丢弃")
                else:
                    print(f"      ✅ {col_names[i]}: 可用 (Mean={mean_val:.0f})")
                    valid_indices.append(i)

            if not valid_indices:
                print("   ❌ 所有通道都坏了！强行使用 Channel 0 抢救...")
                valid_indices = [0]

            # 只取好通道
            selected_data = raw_data[valid_indices, :]

            # --- 信号清洗 ---
            for i in range(selected_data.shape[0]):
                selected_data[i, :] = self.clean_signal(selected_data[i, :])

            # --- 补全/截断到 4 通道 ---
            current_ch = selected_data.shape[0]
            if current_ch < 4:
                print(f"   ⚠️ 通道不足4个 (仅{current_ch}个)，执行复制填充...")
                repeats = (4 // current_ch) + 1
                selected_data = np.tile(selected_data, (repeats, 1))
                selected_data = selected_data[:4, :]
            elif current_ch > 4:
                selected_data = selected_data[:4, :]

            print(f"   -> 最终输入形状: {selected_data.shape}")

            # --- 重采样 (250 -> 1000) ---
            original_fs = 250
            if original_fs != self.model_fs:
                num_samples = int(selected_data.shape[1] * (self.model_fs / original_fs))
                selected_data = signal.resample(selected_data, num_samples, axis=1)

            return selected_data

        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_jsonl_fecg(self, filepath):
        fecg_full = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    if item.get('type') == 'fecg' and 'fecg_signal' in item:
                        fecg_full.extend(item['fecg_signal'])
            data = np.array(fecg_full)
            return data
        except Exception as e:
            print(f"   ⚠️ JSONL 读取失败 (可能没有): {e}")
            return None

    def preprocess_input(self, data, seg_len=1000):
        if data.shape[1] < seg_len: return None
        crop = data[:, :seg_len]

        # Z-score 归一化
        mean = np.mean(crop, axis=1, keepdims=True)
        std = np.std(crop, axis=1, keepdims=True) + 1e-6
        norm_data = (crop - mean) / std

        return torch.from_numpy(norm_data).float()


# ==========================================
# 3. 主程序
# ==========================================
def run_validation():
    # 路径
    txt_dir = "data/comparison"
    jsonl_dir = "data/comparison"
    model_name = "addb_mkf2_improved_0_fecg_diff_0.5"

    c = cfg()
    processor = RobustOpenBCIProcessor(model_fs=c.fs)

    # 加载模型
    print("1. 加载模型...")
    possible_dirs = [os.path.join("results", "model"), os.path.join("resource", "model"), "model"]
    model_dir = next((d for d in possible_dirs if os.path.exists(d)), None)

    if not model_dir:
        print("❌ 找不到模型文件夹")
        return

    try:
        if not os.path.exists(os.path.join(model_dir, model_name + ".pt")):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
            if pt_files: model_name = pt_files[0].replace(".pt", "")

        model = runner.load_model(model_dir=model_dir, model_file=model_name)
        model = model.to(DEVICE)
        print("✅ 模型加载完毕")
    except Exception as e:
        print(f"❌ 模型错误: {e}")
        return

    params = runner.inference_schedule(model)

    # 遍历文件
    print("\n2. 开始验证...")
    files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for filename in files:
        base_name = filename.replace(".txt", "")
        txt_path = os.path.join(txt_dir, filename)
        jsonl_path = os.path.join(jsonl_dir, base_name + ".fecg.jsonl")

        print(f"\n--- 处理: {filename} ---")

        raw_mecg = processor.load_openbci_txt(txt_path)
        baseline_fecg = processor.load_jsonl_fecg(jsonl_path)

        if raw_mecg is None: continue

        demo_len = 2000
        model_input = processor.preprocess_input(raw_mecg, seg_len=demo_len)

        if model_input is None:
            print("❌ 数据太短")
            continue

        # 推理
        model_input = model_input.to(DEVICE)

        print(f"   -> 推理输入形状: {model_input.shape}")
        with torch.no_grad():
            alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar = params
            output = runner.predict(model, model_input,
                                    alpha, beta, alpha_cum, sigmas, T,
                                    c1, c2, c3, delta, delta_bar,
                                    device=DEVICE)

        # 后处理
        ours_1000hz = output[0, :].cpu().numpy()
        ours_200hz = signal.resample(ours_1000hz, int(len(ours_1000hz) / 5))

        # 可视化输入取第0通道
        input_viz = model_input[0, :].cpu().numpy()
        input_viz = signal.resample(input_viz, int(len(input_viz) / 5))

        # 对齐 Baseline
        if baseline_fecg is not None:
            limit = min(len(baseline_fecg), len(ours_200hz))
            baseline_viz = baseline_fecg[:limit]
            ours_viz = ours_200hz[:limit]
            input_viz = input_viz[:limit]

            # 归一化
            baseline_viz = (baseline_viz - np.mean(baseline_viz)) / (np.std(baseline_viz) + 1e-6)
            ours_viz = (ours_viz - np.mean(ours_viz)) / (np.std(ours_viz) + 1e-6)
        else:
            baseline_viz = np.zeros_like(ours_200hz)
            ours_viz = ours_200hz
            input_viz = input_viz[:limit] if 'limit' in locals() else input_viz

        plot_comparison(base_name, input_viz, baseline_viz, ours_viz)


def plot_comparison(name, mecg, baseline, ours):
    time_axis = np.arange(len(mecg)) / 200.0
    Y_LIMIT = [-3, 3]  # 稍微放宽一点视野

    # 创建一个 2列 x 3行 的图
    fig = plt.figure(figsize=(16, 12))

    # === 左边：时域波形 (Time Domain) ===
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time_axis, mecg, 'k', lw=1)
    ax1.set_title(f"Input MECG (Mother + Fetus)", fontsize=10, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(Y_LIMIT)

    ax2 = fig.add_subplot(3, 2, 3)
    ax2.plot(time_axis, baseline, color='#e74c3c', label='Baseline')
    ax2.set_title("Baseline Algorithm FECG", fontsize=10, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.set_ylim(Y_LIMIT)

    ax3 = fig.add_subplot(3, 2, 5)
    ax3.plot(time_axis, ours, color='#27ae60', label='Ours')
    ax3.set_title("Ours (DIFF-FECG)", fontsize=10, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend()
    ax3.set_ylim(Y_LIMIT)

    # === 右边：频域功率谱 (Frequency Domain) ===
    # 辅助函数：计算 PSD
    def plot_psd(ax, data, fs=200, color='blue'):
        f, Pxx = signal.welch(data, fs, nperseg=1024)
        ax.semilogy(f, Pxx, color=color, lw=1.5)
        ax.set_xlim(0, 5)  # 只看 0-5Hz (心率范围)
        ax.grid(True, which='both', alpha=0.3)
        # 找最大峰值频率
        peak_freq = f[np.argmax(Pxx)]
        bpm = peak_freq * 60
        ax.text(0.5, 0.9, f"Peak: {peak_freq:.2f}Hz ({bpm:.0f} BPM)",
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax4 = fig.add_subplot(3, 2, 2)
    plot_psd(ax4, mecg, color='black')
    ax4.set_title("Input Spectrum (Should match Mother BPM)", fontsize=9)

    ax5 = fig.add_subplot(3, 2, 4)
    plot_psd(ax5, baseline, color='#e74c3c')
    ax5.set_title("Baseline Spectrum", fontsize=9)

    ax6 = fig.add_subplot(3, 2, 6)
    plot_psd(ax6, ours, color='#27ae60')
    ax6.set_title("Ours Spectrum (Check BPM!)", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_validation()