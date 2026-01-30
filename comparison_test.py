import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib.ticker import MultipleLocator

# ================= 配置区域 =================
INPUT_DIR = "data/comparison"
OUTPUT_DIR = "results/workTogether"
FS = 200  # 采样率
DISPLAY_SECONDS = 5  # 显示时长


# ===========================================

def apply_filters(signal, fs):
    """
    对信号应用陷波滤波和带通滤波
    """
    # 1. 50Hz 陷波滤波器 (去除工频噪声)
    f0 = 50.0  # 频率
    Q = 30.0  # 品质因数
    b_notch, a_notch = iirnotch(f0, Q, fs)
    signal_notched = filtfilt(b_notch, a_notch, signal)

    # 2. 带通滤波器 (0.5Hz - 45Hz, 提取典型心电/脑电范围)
    lowcut = 0.5
    highcut = 45.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = butter(4, [low, high], btype='band')
    signal_filtered = filtfilt(b_band, a_band, signal_notched)

    return signal_filtered


def plot_cleaned_segment(filepath, save_dir):
    filename = os.path.basename(filepath)
    print(f"正在处理并滤波: {filename} ...")

    try:
        # 1. 读取数据
        df = pd.read_csv(filepath, comment='%', header=0, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # 2. 识别前 6 个 EXG 通道
        exg_cols = [c for c in df.columns if 'EXG Channel' in c]
        if not exg_cols:
            exg_cols = df.columns[1:7]
        else:
            exg_cols = exg_cols[:6]

        # 3. 寻找有效数据起始点
        data_matrix = df[exg_cols].values
        non_zero_indices = np.where(np.any(data_matrix != 0, axis=1))[0]

        if len(non_zero_indices) == 0:
            print(f"   ⚠️ 文件 {filename} 无有效数据，跳过。")
            return

        start_idx = non_zero_indices[0]

        # 为了让滤波器稳定，我们多截取一点数据用于处理，最后再切掉边缘
        pad = FS * 1  # 1秒缓冲区
        process_start = max(0, start_idx - pad)
        process_end = start_idx + (DISPLAY_SECONDS * FS) + pad

        # 4. 创建画布
        num_channels = len(exg_cols)
        fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(16, 2 * num_channels), sharex=True)
        if num_channels == 1: axes = [axes]

        fig.suptitle(f"Cleaned ECG (Notch 50Hz + Bandpass): {filename}", fontsize=15, fontweight='bold', y=0.99)

        # 5. 循环滤波并绘制
        for i in range(num_channels):
            ax = axes[i]

            # 获取原始片段
            raw_sig = data_matrix[process_start:process_end, i]

            # 去均值并应用滤波器
            centered_sig = raw_sig - np.mean(raw_sig)
            filtered_sig = apply_filters(centered_sig, FS)

            # 截回我们真正要看的 5 秒部分 (去掉缓冲区)
            display_start_in_seg = start_idx - process_start
            final_plot_data = filtered_sig[display_start_in_seg: display_start_in_seg + (DISPLAY_SECONDS * FS)]

            time_axis = np.arange(len(final_plot_data)) / FS

            # 绘制波形 (建议用红色或蓝色以区分原始数据)
            ax.plot(time_axis, final_plot_data, color='#c0392b', linewidth=1.0)

            # 设定要求：±200uV，1s网格
            ax.set_ylim(-200, 200)
            ax.set_xlim(0, DISPLAY_SECONDS)
            ax.xaxis.set_major_locator(MultipleLocator(1))

            ax.set_ylabel(f"Ch {i}\n(uV)", rotation=0, labelpad=25, fontsize=10, fontweight='bold')
            ax.grid(True, which='major', linestyle='-', alpha=0.7, color='#bdc3c7')
            ax.legend([f"{exg_cols[i]} [FILTERED]"], loc='upper right', frameon=True, fontsize=8)

        axes[-1].set_xlabel("Time (s)", fontsize=12)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # 6. 保存
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{filename.replace('.txt', '')}_Cleaned_5s.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"   ✅ 处理完成：{save_path}")

    except Exception as e:
        print(f"   ❌ 出错: {e}")


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录: {INPUT_DIR}")
        return
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    for f in files:
        plot_cleaned_segment(os.path.join(INPUT_DIR, f), OUTPUT_DIR)
    print("🎉 任务结束，请查看 results 文件夹中的图片。")


if __name__ == "__main__":
    main()