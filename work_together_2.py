import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MultipleLocator

# ================= 配置区域 =================
INPUT_DIR = "data/comparison"
OUTPUT_DIR = "results/workTogether2"
FS = 200
DISPLAY_SECONDS = 5
Y_LIMIT = 200  # 目标范围

# 🔴🔴🔴【在这里修改起始时间】🔴🔴🔴
MANUAL_START_TIME = 136


# ===========================================

def remove_baseline_only(signal):
    """
    全局去基线 (去直流分量)
    """
    return signal - np.mean(signal)


def find_stable_start_index(data_matrix, window_size, limit=190):
    total_len = len(data_matrix)
    step = int(FS * 0.5)

    for start in range(0, total_len - window_size, step):
        window = data_matrix[start: start + window_size]
        if np.max(np.abs(window)) < limit:
            return start

    return min(int(FS * 3), total_len - window_size)


def plot_raw_baseline_removed(filepath, save_dir):
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

        # 3. 处理全段数据（这里先做一次全局去平均值）
        all_processed_list = []
        for col in exg_cols:
            raw_sig = df[col].values
            sig_centered = remove_baseline_only(raw_sig)
            all_processed_list.append(sig_centered)

        full_data_matrix = np.array(all_processed_list).T

        # 4. 确定波段的起始点
        window_pts = DISPLAY_SECONDS * FS

        if MANUAL_START_TIME is not None:
            best_start = int(MANUAL_START_TIME * FS)
            print(f"   🛠️ 模式: 手动指定起始点")
            if best_start + window_pts > total_rows:
                best_start = max(0, total_rows - window_pts)
        else:
            best_start = find_stable_start_index(full_data_matrix, window_pts, limit=Y_LIMIT)
            print(f"   🤖 模式: 自动寻找稳定段")

        print(f"   🎯 最终截取起点: 第 {best_start / FS:.2f} 秒")

        # 5. 绘图
        fig, axes = plt.subplots(nrows=len(exg_cols), ncols=1, figsize=(16, 12), sharex=True)

        title_mode = f"Manual Start @ {best_start / FS:.2f}s" if MANUAL_START_TIME is not None else "Auto-Detected Stable Segment"
        fig.suptitle(
            f"RAW Data (Local Mean Removed) | {title_mode} | {filename}\nTotal Duration: {total_duration_sec:.2f}s",
            fontsize=14, fontweight='bold')

        for i, col_name in enumerate(exg_cols):
            ax = axes[i]
            # 截取选定的段
            display_data = full_data_matrix[best_start: best_start + window_pts, i]

            # 👇👇👇【新增重点】👇👇👇
            # 针对这短短的 5 秒再次去平均值 (Local De-meaning)
            # 这样可以确保波形在图表中绝对居中，不会因为基线漂移而偏离 0 刻度
            display_data = display_data - np.mean(display_data)
            # 👆👆👆【新增结束】👆👆👆

            time_axis = np.arange(len(display_data)) / FS

            # 绘图
            ax.plot(time_axis, display_data, color='#333333', linewidth=0.8)

            # 设置坐标轴
            ax.set_ylim(-Y_LIMIT, Y_LIMIT)
            ax.set_xlim(0, DISPLAY_SECONDS)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.grid(True, which='major', alpha=0.5, linestyle='--')
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
    if not os.path.exists(INPUT_DIR):
        print(f"文件夹 {INPUT_DIR} 不存在")
        return
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    for f in files:
        plot_raw_baseline_removed(os.path.join(INPUT_DIR, f), OUTPUT_DIR)


if __name__ == "__main__":
    main()