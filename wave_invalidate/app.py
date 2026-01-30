import sys
import os
import time

# --- 路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # 引入信号处理库
from fecg_service import FecgInferenceService

# ============================================
# 1. 网页配置
# ============================================
st.set_page_config(
    page_title="DIFF-FECG 临床监护台",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        h1 {margin-bottom: 0.5rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🏥 DIFF-FECG 临床监护台 (Pro)")


# ============================================
# 2. 初始化与服务加载
# ============================================
@st.cache_resource
def load_service():
    return FecgInferenceService(model_name="addb_mkf2_improved_0_fecg_diff_0.5")


try:
    service = load_service()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False


# ============================================
# 3. 核心信号处理函数 (满足你的4点要求)
# ============================================
def advanced_signal_processing(data, fs=250):
    """
    执行严格的信号预处理流程
    """
    # [要求1] 剔除 > 100k 的坏点 (Clipping)
    # 将超过 +/- 100,000 的值强制设为边界值，防止滤波器发散
    data = np.clip(data, -100000, 100000)

    # [要求2] 组合滤波
    # A. 50Hz 陷波 (去除工频干扰)
    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    data = signal.filtfilt(b_notch, a_notch, data)

    # B. 5-50Hz 带通 (去除基线漂移和高频肌电)
    sos = signal.butter(4, [5, 50], btype='bandpass', fs=fs, output='sos')
    data = signal.sosfiltfilt(sos, data)

    return data


def calculate_robust_amplitude(data):
    """
    [要求4] 忽略前1%和后1%的极值进行缩放计算
    """
    if len(data) == 0: return 1.0
    # 计算第 1 百分位 和 第 99 百分位
    p1, p99 = np.percentile(data, [1, 99])
    # 估算幅度范围
    amplitude = (p99 - p1) / 2.0
    # 防止除以0
    return amplitude if amplitude > 1e-6 else 1.0


# ============================================
# 4. 全量预计算逻辑
# ============================================
def preprocess_all_channels(df, cols):
    results = []
    total = len(cols)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, col in enumerate(cols):
        status_text.text(f"正在进行高级处理: 通道 {col} ({i + 1}/{total})...")

        # 1. 获取原始数据
        raw_full = df[col].values

        # 2. 执行高级滤波 (坏点去除 + 陷波 + 带通)
        # 注意：这会改变 raw_viz_full 的形态，使其变得非常干净
        clean_raw_full = advanced_signal_processing(raw_full)

        # 3. 计算鲁棒幅度 (Robust Amplitude)
        # 用于将 AI 重建的归一化波形拉伸回真实电压范围
        robust_amp = calculate_robust_amplitude(clean_raw_full)

        # 4. AI 推理 (使用原始数据还是滤波数据？)
        # 这是一个策略选择。通常 DIFF-FECG 模型训练时包含噪声。
        # 这里我们把 原始数据 喂给 AI，看它能不能处理。
        # 如果你想让 AI 效果更好，也可以喂 clean_raw_full
        fecg_out = service.process_single_channel(raw_full)

        # 5. 恢复幅度 (使用鲁棒缩放因子)
        fecg_viz_full = fecg_out * robust_amp

        results.append({
            "name": col,
            "raw": clean_raw_full,  # 存入清洗后的数据用于显示
            "fecg": fecg_viz_full
        })
        progress_bar.progress((i + 1) / total)

    status_text.empty()
    progress_bar.empty()
    return results


# ============================================
# 5. 侧边栏与主界面
# ============================================
st.sidebar.header("📁 1. 数据导入")
uploaded_file = st.sidebar.file_uploader("上传 OpenBCI 数据", type=["txt"])

st.sidebar.header("⚙️ 2. 显示设置")
y_limit = st.sidebar.slider("纵坐标范围 (uV)", 20, 300, 100, step=10)  # 范围改小点，因为去除了噪声
window_size = st.sidebar.slider("窗口宽度 (秒)", 2, 10, 5)
play_speed = st.sidebar.select_slider("回放速度", options=["慢", "中", "快"], value="中")
speed_map = {"慢": 0.05, "中": 0.1, "快": 0.25}  # 稍微调慢一点以便观察细节
step_sec = speed_map[play_speed]

if uploaded_file:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.processed_data = None
        st.session_state.current_file = uploaded_file.name
        st.session_state.is_playing = False

    if st.session_state.processed_data is None:
        try:
            df = pd.read_csv(uploaded_file, comment='%', header=0, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            exg_cols = [c for c in df.columns if 'EXG Channel' in c]
            if not exg_cols: exg_cols = df.columns[1:9] if df.shape[1] >= 9 else []
            target_cols = exg_cols[:6]

            st.info(f"文件已加载，准备进行 {len(target_cols)} 通道的高级信号处理。")
            if st.button("🚀 开始 AI 分析 (含滤波与去噪)"):
                with st.spinner("正在执行: 坏点剔除 -> 陷波 -> 带通 -> 鲁棒缩放 -> AI 推理..."):
                    data_package = preprocess_all_channels(df, target_cols)
                    st.session_state.processed_data = data_package
                    st.rerun()
        except Exception as e:
            st.error(f"读取错误: {e}")
            st.stop()

    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        total_points = len(data[0]['raw'])
        total_seconds = total_points / 250.0

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("▶️ 播放 / ⏸️ 暂停", type="primary"):
                st.session_state.is_playing = not st.session_state.is_playing
        with col2:
            start_time = st.slider("时间进度", 0.0, total_seconds - window_size, 0.0, 0.1)

        chart_placeholder = st.empty()


        def draw_frame(current_start_time):
            start_idx = int(current_start_time * 250)
            end_idx = int((current_start_time + window_size) * 250)

            fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(10, 18), sharex=True)
            t_axis = np.linspace(current_start_time, current_start_time + window_size, end_idx - start_idx)

            for i, ch_data in enumerate(data):
                raw_seg = ch_data['raw'][start_idx:end_idx]
                fecg_seg = ch_data['fecg'][start_idx:end_idx]

                # 安全对齐
                min_len = min(len(t_axis), len(raw_seg), len(fecg_seg))
                if min_len == 0: continue

                t_seg = t_axis[:min_len]
                raw_seg = raw_seg[:min_len]
                fecg_seg = fecg_seg[:min_len]

                # [要求3] 减去当前窗口均值 (Real-time De-mean)
                # 这一步确保波形永远垂直居中
                raw_seg = raw_seg - np.mean(raw_seg)
                fecg_seg = fecg_seg - np.mean(fecg_seg)

                # 绘图 - 原始
                ax_raw = axes[i * 2]
                ax_raw.plot(t_seg, raw_seg, 'k', lw=1)
                ax_raw.set_ylim([-y_limit, y_limit])
                ax_raw.set_yticks([])
                ax_raw.text(0.01, 0.8, f"Ch{i} Cleaned Input", transform=ax_raw.transAxes, fontsize=8,
                            fontweight='bold')
                ax_raw.grid(alpha=0.2, linestyle='--')

                # 绘图 - 重建
                ax_fecg = axes[i * 2 + 1]
                ax_fecg.plot(t_seg, fecg_seg, '#27ae60', lw=1.2)
                ax_fecg.set_ylim([-y_limit, y_limit])  # 同样应用限制
                ax_fecg.set_yticks([])
                ax_fecg.text(0.01, 0.8, f"Ch{i} FECG Output", transform=ax_fecg.transAxes, fontsize=8,
                             fontweight='bold', color='green')
                ax_fecg.grid(alpha=0.2, linestyle='--')

            axes[-1].set_xlabel("Time (s)")
            axes[-1].set_xlim(current_start_time, current_start_time + window_size)
            plt.tight_layout(pad=0.5, h_pad=0.1)
            return fig


        if st.session_state.is_playing:
            curr_t = start_time
            while curr_t < total_seconds - window_size:
                if not st.session_state.is_playing: break

                fig = draw_frame(curr_t)
                chart_placeholder.pyplot(fig)
                plt.close(fig)

                curr_t += step_sec
                time.sleep(0.01)
        else:
            fig = draw_frame(start_time)
            chart_placeholder.pyplot(fig)
            plt.close(fig)

else:
    st.info("👈 请上传数据文件以开始")

    #test