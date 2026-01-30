import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ================= 配置区域 =================
# 输入文件夹：存放你那 7 个 txt 的地方
INPUT_DIR = "data/comparison"
# 输出文件夹：图片保存的地方
OUTPUT_DIR = "results/workTogether2"
# 采样率
FS = 250


# ===========================================

def plot_file_channels(filepath, save_dir):
    filename = os.path.basename(filepath)
    print(f"📈 正在绘图: {filename} ...")

    try:
        # 1. 读取数据
        df = pd.read_csv(filepath, comment='%', header=0, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # 2. 寻找 EXG 通道
        exg_cols = [c for c in df.columns if 'EXG Channel' in c]
        # 如果找不到标准列名，尝试取第 1-8 列
        if not exg_cols:
            if df.shape[1] >= 9:
                exg_cols = df.columns[1:9]
            else:
                print(f"   ⚠️ 跳过: 找不到数据列")
                return

        # 3. 创建画布 (8行1列)
        num_channels = len(exg_cols)
        # 动态调整高度：每个通道给 2 英寸高
        fig, axes = plt.subplots(nrows=num_channels, ncols=1,
                                 figsize=(15, 2 * num_channels),
                                 sharex=True)

        # 处理单通道的特殊情况
        if num_channels == 1: axes = [axes]

        fig.suptitle(f"Raw Data Inspection: {filename}", fontsize=16, fontweight='bold', y=0.99)

        # 时间轴
        num_samples = len(df)
        time_axis = np.arange(num_samples) / FS

        # 4. 逐个通道绘图
        for i, col in enumerate(exg_cols):
            ax = axes[i]
            raw_data = df[col].values

            # --- 关键判断：是否饱和 ---
            # 只要绝对值的最大值超过 180000，就认为是饱和
            # 或者均值极其异常
            is_railed = np.max(np.abs(raw_data)) > 180000

            if is_railed:
                # === 饱和通道处理 ===
                # 1. 画原始数据 (不减均值)，让用户看到真实的 -187500
                # 2. 红色，线条稍微加粗
                ax.plot(time_axis, raw_data, color='#e74c3c', linewidth=1.5, label='RAILED (Raw Value)')

                # 在图中间写个大大的 RAILED
                ax.text(0.5, 0.5, f"RAILED / SATURATED\n(Value: {np.mean(raw_data):.1f})",
                        transform=ax.transAxes, ha='center', va='center',
                        color='red', fontsize=14, fontweight='bold', alpha=0.3)

                # 强制 Y 轴范围显示出这个巨大的数值
                # 稍微给一点余量，防止线贴着边
                mean_val = np.mean(raw_data)
                ax.set_ylim(mean_val - 1000, mean_val + 1000)

            else:
                # === 正常通道处理 ===
                # 1. 去均值 (Centered)，方便看波形
                # 2. 蓝色，细线
                data_centered = raw_data - np.mean(raw_data)
                ax.plot(time_axis, data_centered, color='#2980b9', linewidth=0.8, label='Normal (Centered)')

                # 添加网格
                ax.grid(True, linestyle=':', alpha=0.6)

            # 设置标签
            ax.set_ylabel(f"Ch {i}\n({col})", rotation=0, labelpad=40, fontsize=9, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)

        # 底部 X 轴
        axes[-1].set_xlabel("Time (s)", fontsize=12)
        axes[-1].set_xlim(0, time_axis[-1])

        # 调整布局
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # 5. 保存
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{filename.replace('.txt', '')}_full.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"   ✅ 保存成功: {save_path}")

    except Exception as e:
        print(f"   ❌ 处理失败 {filename}: {e}")


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入文件夹: {INPUT_DIR}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"📂 扫描到 {len(files)} 个 txt 文件")

    for f in files:
        file_path = os.path.join(INPUT_DIR, f)
        plot_file_channels(file_path, OUTPUT_DIR)

    print(f"\n🎉 全部绘图完成！请查看文件夹: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()