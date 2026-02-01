import numpy as np
import pandas as pd
from scipy import signal
import os


def parse_and_clean_txt(file_path):
    """
    1. 解析 TXT (参考 parseTxtFull)
    2. 去坏点 (参考 if Math.abs(val) < 100000)
    """
    # 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fs = 250  # 默认采样率
    exg_indices = {}  # 通道索引映射
    valid_channels = [0, 1, 2, 3, 4, 5]  # HTML 中定义的 VALID_CHANNELS

    # 初始化数据容器
    raw_data = {ch: [] for ch in valid_channels}

    header_passed = False

    for line in lines:
        line = line.strip()
        if not line: continue

        # 解析采样率
        if line.startswith('%'):
            if 'Sample Rate' in line:
                try:
                    fs = int(line.split('=')[1].strip().split()[0])
                except:
                    pass
            continue

        # 解析表头
        if not header_passed:
            parts = [p.strip() for p in line.split(',')]
            for idx, col_name in enumerate(parts):
                # 匹配 "EXG Channel 0", "EXG Channel 1" ...
                if 'EXG Channel' in col_name:
                    try:
                        ch_num = int(col_name.split('Channel')[1].strip())
                        if ch_num in valid_channels:
                            exg_indices[ch_num] = idx
                    except:
                        pass
            header_passed = True
            continue

        # 解析数据行 & 去坏点逻辑
        parts = line.split(',')
        for ch_num in valid_channels:
            if ch_num not in exg_indices:
                continue

            col_idx = exg_indices[ch_num]
            if col_idx >= len(parts):
                val = 0.0
            else:
                try:
                    val = float(parts[col_idx])
                except:
                    val = 0.0

            # --- HTML 逻辑复刻: if (isFinite(val) && Math.abs(val) < 100000) ---
            # 如果是坏点，则沿用上一个点的值 (Sample-and-Hold)
            current_list = raw_data[ch_num]
            if np.isfinite(val) and abs(val) < 100000:
                current_list.append(val)
            else:
                # 使用上一个值，若没有上一个值则补0
                last_val = current_list[-1] if len(current_list) > 0 else 0.0
                current_list.append(last_val)

    # 转换为 numpy 数组 (Channels x Time)
    # 确保所有通道长度一致
    min_len = min([len(raw_data[ch]) for ch in raw_data if raw_data[ch]])
    data_matrix = []
    for ch in valid_channels:
        if ch in raw_data and raw_data[ch]:
            data_matrix.append(raw_data[ch][:min_len])
        else:
            # 如果某通道没数据，补全0
            data_matrix.append(np.zeros(min_len))

    return np.array(data_matrix), fs


def apply_html_filters(data, fs):
    """
    复刻 viewer.html 中的 computeFilteredChannels 逻辑
    顺序: Bandpass (High 5Hz -> Low 50Hz) -> Notch 50Hz -> Notch 60Hz
    """
    processed = data.copy()

    # 1. Highpass 5Hz (Butterworth 2nd order)
    # HTML: designBiquad(fs, 'highpass', 5, 0.707)
    b_hp, a_hp = signal.butter(2, 5, btype='highpass', fs=fs)
    processed = signal.lfilter(b_hp, a_hp, processed, axis=1)

    # 2. Lowpass 50Hz (Butterworth 2nd order)
    # HTML: designBiquad(fs, 'lowpass', 50, 0.707)
    b_lp, a_lp = signal.butter(2, 50, btype='lowpass', fs=fs)
    processed = signal.lfilter(b_lp, a_lp, processed, axis=1)

    # 3. Notch 50Hz (Q=30)
    # HTML: designBiquad(fs, 'notch', 50, 30)
    b_n50, a_n50 = signal.iirnotch(50, 30, fs=fs)
    processed = signal.lfilter(b_n50, a_n50, processed, axis=1)

    # 4. Notch 60Hz (Q=30)
    # HTML: designBiquad(fs, 'notch', 60, 30)
    b_n60, a_n60 = signal.iirnotch(60, 30, fs=fs)
    processed = signal.lfilter(b_n60, a_n60, processed, axis=1)

    return processed


def process_like_viewer(txt_path):
    print(f"处理文件: {txt_path}")

    # 步骤 1: 解析 + 去坏点 (Sample-and-Hold)
    data, fs = parse_and_clean_txt(txt_path)

    if data.size == 0:
        print("无有效数据")
        return None

    # 步骤 2: 滤波 (5-50Hz + 50/60Hz Notch)
    data_filtered = apply_html_filters(data, fs)

    # 步骤 3: 去直流 (Demean)
    # HTML 中是渲染时对当前窗口去均值，这里对整段信号去均值
    data_final = data_filtered - np.mean(data_filtered, axis=1, keepdims=True)

    return data_final, fs


# ================= 使用示例 =================
if __name__ == "__main__":
    # 替换成你的文件路径
    input_file = "raw_data/LQBCI-RAW-2026-01-27_14-56-21.txt"

    if os.path.exists(input_file):
        processed_data, fs = process_like_viewer(input_file)

        print(f"处理完成! 数据形状: {processed_data.shape}, 采样率: {fs}")

        # 保存为 npy 供后续使用
        np.save("processed_like_html.npy", processed_data)
        print("已保存为 processed_like_html.npy")
    else:
        print(f"找不到文件: {input_file}")