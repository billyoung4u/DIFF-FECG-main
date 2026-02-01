import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# ================= 配置区域 =================
INPUT_FOLDER = "raw_data"  # 把你的原始 txt 放在这个文件夹
OUTPUT_FOLDER = "processed_data"  # 处理结果会保存在这里
TARGET_FS = 1000  # ADDB 的标准采样率
ORIGIN_FS = 250  # OpenBCI 的原始采样率


# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_txt_data(file_path):
    """读取 OpenBCI 格式的 TXT 文件"""
    try:
        # 跳过注释行，读取 CSV
        df = pd.read_csv(file_path, comment='%', header=0, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # 自动寻找 EXG 通道
        channels = [c for c in df.columns if 'EXG Channel' in c]
        if not channels:
            # 如果找不到 EXG，尝试取第 1-8 列
            channels = df.columns[1:9] if df.shape[1] >= 9 else []

        # 限制只取前 4 个通道 (ADDB 通常是 4 通道)
        target_cols = channels[:4]

        data_matrix = df[target_cols].values.T  # 转置为 (Channels, Length)
        return data_matrix, list(target_cols)
    except Exception as e:
        print(f"❌ 读取失败 {file_path}: {e}")
        return None, None


def align_to_addb_standard(raw_data, fs_old, fs_new):
    """
    核心函数：将信号对齐到 ADDB 标准
    1. 坏点剔除
    2. 50Hz 陷波
    3. 重采样
    4. [新增] 7.5-75Hz 带通滤波
    5. Z-Score 标准化
    """
    n_channels, n_length = raw_data.shape

    # 1. 坏点剔除 (Clipping)
    # 原始数据可能包含极大值，先限制在合理范围
    data = np.clip(raw_data, -100000, 100000)

    # 2. 50Hz 陷波 (Notch Filter)
    # 在原始采样率 (250Hz) 下去除工频干扰
    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs_old)
    data = signal.filtfilt(b_notch, a_notch, data, axis=1)

    # 3. 重采样 (Resampling)
    # 250Hz -> 1000Hz
    target_length = int(n_length * fs_new / fs_old)
    data_resampled = signal.resample(data, target_length, axis=1)

    # =========================================================
    # 🔥 [新增操作] 4. 带通滤波 (7.5Hz - 75Hz)
    # =========================================================
    # 我们在重采样后的 fs_new (1000Hz) 下进行滤波
    # 使用 sosfiltfilt 保证零相位偏移（不改变波峰位置）
    sos_bp = signal.butter(4, [7.5, 75], btype='bandpass', fs=fs_new, output='sos')
    data_filtered = signal.sosfiltfilt(sos_bp, data_resampled, axis=1)

    # 5. Z-Score 标准化 (Normalization)
    # axis=1 表示对每个通道独立归一化
    # 此时的数据已经是 1000Hz 且经过了 7.5-75Hz 滤波
    mean = np.mean(data_filtered, axis=1, keepdims=True)
    std = np.std(data_filtered, axis=1, keepdims=True) + 1e-6
    data_normalized = (data_filtered - mean) / std

    return data_normalized


def plot_comparison(raw, processed, filename, save_dir):
    """画对比图：原始 vs 处理后"""
    plt.figure(figsize=(12, 6))

    # 画第一个通道即可
    plt.subplot(2, 1, 1)
    plt.plot(raw[0, :1000], color='gray', label='Raw (250Hz)')
    plt.title(f"{filename} - Raw Channel 0 (First 4s)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 1, 2)
    # 处理后的数据是 1000Hz，所以 4s 是 4000 个点
    plt.plot(processed[0, :4000], color='blue', lw=0.8, label='Processed (7.5-75Hz + Norm)')
    plt.title(f"Processed to ADDB Standard (1000Hz)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_check.png"))
    plt.close()


def main():
    ensure_dir(INPUT_FOLDER)
    ensure_dir(OUTPUT_FOLDER)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]

    if not files:
        print(f"⚠️  文件夹 '{INPUT_FOLDER}' 是空的，请放入 .txt 文件！")
        return

    print(f"🚀 开始处理，共发现 {len(files)} 个文件...")

    for i, file_name in enumerate(files):
        print(f"[{i + 1}/{len(files)}] 正在处理: {file_name} ...")

        # 1. 加载
        file_path = os.path.join(INPUT_FOLDER, file_name)
        raw_data, ch_names = load_txt_data(file_path)

        if raw_data is None: continue

        # 2. 对齐处理 (含新增的带通滤波)
        processed_data = align_to_addb_standard(raw_data, ORIGIN_FS, TARGET_FS)

        # 3. 导出保存
        base_name = os.path.splitext(file_name)[0]

        # 保存为 .npy (用于 python/pytorch)
        npy_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_addb.npy")
        np.save(npy_path, processed_data)

        # 保存为 .csv (用于 excel/matlab)
        csv_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_addb.csv")
        # 如果原始通道名少于实际数据通道数（因为前面可能有截取），这里做个安全处理
        safe_ch_names = ch_names if len(ch_names) == processed_data.shape[0] else [f"Ch{j}" for j in
                                                                                   range(processed_data.shape[0])]
        df_out = pd.DataFrame(processed_data.T, columns=safe_ch_names)
        df_out.to_csv(csv_path, index=False)

        # 4. 画对比图
        plot_comparison(raw_data, processed_data, base_name, OUTPUT_FOLDER)

    print(f"\n✅ 全部完成！结果已保存在 '{OUTPUT_FOLDER}' 文件夹。")


if __name__ == "__main__":
    main()