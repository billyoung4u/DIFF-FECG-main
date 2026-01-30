import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib.ticker import MultipleLocator

# ================= 配置区域 =================
INPUT_DIR = "data/comparison"
OUTPUT_DIR = "results/workTogether"
FS = 200
DISPLAY_SECONDS = 5
Y_LIMIT = 200

# 🔴🔴🔴【在这里修改起始时间】🔴🔴🔴
# 设置为具体的数字 (例如 0, 10.5, 60) 来强制指定起始秒数。
# 设置为 None (注意首字母大写)，则启用之前的“自动寻找稳定波段”功能。
MANUAL_START_TIME = 136


# ===========================================

def apply_advanced_filters(signal, fs):
    """ 高级滤波：50Hz陷波 + 1-30Hz带通 (保持不变) """
    b_notch, a_notch = iirnotch(50.0, 30.0, fs)
    signal_notched = filtfilt(b_notch, a_notch, signal)

    nyq = 0.5 * fs
    low, high = 1.0 / nyq, 30.0 / nyq
    b_band, a_band = butter(4, [low, high], btype='band')
    return filtfilt(b_band, a_band, signal_notched)


def find_stable_start_index(filtered_data_matrix, window_size, limit=190):
    """ (保持不变) 自动寻找稳定窗口 """
    total_len = len(filtered_data_matrix)
    step = int(FS * 0.5)

    for start in range(0, total_len - window_size, step):
        window = filtered_data_matrix[start: start + window_size]
        if np.max(np.abs(window)) < limit:
            return start
    return min(int(FS * 3), total_len - window_size)


def plot_final_cleaned_data(filepath, save_dir):
    filename = os.path.basename(filepath)

    try:
        # 1. 读取数据
        df = pd.read_csv(filepath, comment='%', header=0, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # 计算总时长
        total_rows = len(df)
        total_duration_sec = total_rows / FS
        print(f"\n文件: {filename}")
        print(f"   ⏱️ 文件总时长: {total_duration_sec:.2f} 秒")

        # 2. 识别通道
        exg_cols = [c for c in df.columns if 'EXG Channel' in c][:6]
        if not exg_cols: exg_cols = df.columns[1:7]

        # 3. 对全段进行滤波
        all_filtered_list = []
        for col in exg_cols:
            raw_sig = df[col].values
            # 去均值后滤波
            sig_clean = apply_advanced_filters(raw_sig - np.mean(raw_sig), FS)
            all_filtered_list.append(sig_clean)

        full_filtered_matrix = np.array(all_filtered_list).T

        # 4. 确定波段的起始点 (修改了这里)
        window_pts = DISPLAY_SECONDS * FS

        # ----------- ⏰ 修改逻辑开始 -----------
        if MANUAL_START_TIME is not None:
            # 手动模式
            best_start = int(MANUAL_START_TIME * FS)
            print(f"   🛠️ 模式: 手动指定起始点 -> 第 {MANUAL_START_TIME} 秒")

            # 防止超出文件范围
            if best_start + window_pts > total_rows:
                print(f"   ⚠️ 警告: 指定时间超出文件长度，自动调整到末尾前5秒")
                best_start = max(0, total_rows - window_pts)
        else:
            # 自动模式
            best_start = find_stable_start_index(full_filtered_matrix, window_pts, limit=Y_LIMIT - 10)
            print(f"   🤖 模式: 自动截取稳定波段")
        # ----------- ⏰ 修改逻辑结束 -----------

        print(f"   🎯 最终截取起点: 第 {best_start / FS:.2f} 秒")

        # 5. 绘图
        fig, axes = plt.subplots(nrows=len(exg_cols), ncols=1, figsize=(16, 12), sharex=True)

        mode_str = f"Manual Start @ {best_start / FS}s" if MANUAL_START_TIME is not None else "Auto-Stable"
        fig.suptitle(f"Filtered Data | {mode_str} | {filename}\nTotal Duration: {total_duration_sec:.2f}s",
                     fontsize=14, fontweight='bold')

        for i, col_name in enumerate(exg_cols):
            ax = axes[i]
            # 截取选定的段
            display_data = full_filtered_matrix[best_start: best_start + window_pts, i]
            time_axis = np.arange(len(display_data)) / FS

            ax.plot(time_axis, display_data, color='#2c3e50', linewidth=1.0)
            ax.set_ylim(-Y_LIMIT, Y_LIMIT)
            ax.set_xlim(0, DISPLAY_SECONDS)
            ax.grid(True, which='major', alpha=0.5)
            ax.set_ylabel(f"Ch {i}", rotation=0, labelpad=20)

        axes[-1].set_xlabel(f"Time (s) [Segment from {best_start / FS:.2f}s to {(best_start + window_pts) / FS:.2f}s]")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{filename.replace('.txt', '')}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"   ✅ 图片已保存: {save_path}")

    except Exception as e:
        print(f"   ❌ 处理出错: {e}")


def main():
    if not os.path.exists(INPUT_DIR): return
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    for f in files:
        plot_final_cleaned_data(os.path.join(INPUT_DIR, f), OUTPUT_DIR)


if __name__ == "__main__":
    main()