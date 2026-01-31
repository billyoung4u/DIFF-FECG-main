import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# --- 路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

from inference_core import InferenceCore
from data_stream import TxtDataStream, EdfFileStreamer

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="DIFF-FECG 临床监护系统",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# CSS 优化：让图表区域背景更干净，文字更清晰，减少顶部空白
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;} 
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    /* 隐藏默认菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 优化指标卡片样式 */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 初始化状态 (Session State)
# ==========================================
if 'core' not in st.session_state:
    with st.spinner("正在启动 AI 引擎..."):
        st.session_state.core = InferenceCore()

if 'stream' not in st.session_state:
    st.session_state.stream = None

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if 'current_time' not in st.session_state:
    st.session_state.current_time = 0.0

# ==========================================
# 3. 侧边栏：设置与控制
# ==========================================
st.sidebar.title("🎛️ 监护控制台")
# 修改 file_uploader 支持两种格式
uploaded_file = st.sidebar.file_uploader("📂 加载病例数据", type=['txt', 'edf'])

# 文件加载逻辑
if uploaded_file:
    last_file = st.session_state.get('last_filename', None)
    if last_file != uploaded_file.name:
        # 🔥 根据后缀名判断使用哪个加载器
        if uploaded_file.name.lower().endswith('.edf'):
            try:
                st.session_state.stream = EdfFileStreamer(uploaded_file)
                st.sidebar.success(f"EDF 文件已加载: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"EDF 读取失败: {e}")
                st.stop()
        else:
            # 默认为 TXT
            st.session_state.stream = TxtDataStream(uploaded_file)
            st.sidebar.success(f"TXT 文件已加载: {uploaded_file.name}")

        st.session_state.last_filename = uploaded_file.name
        st.session_state.current_time = 0.0
        st.session_state.is_running = False
        st.sidebar.success(f"已就绪: {uploaded_file.name}")

st.sidebar.divider()

# --- [新功能 1] 通道选择下拉菜单 ---
channel_options = [f"Channel {i}" for i in range(6)]
selected_channel_str = st.sidebar.selectbox(
    "📺 选择监测通道",
    channel_options,
    index=0,
    help="选择要详细分析的导联通道"
)
# 解析出通道索引 (0-5)
selected_ch_idx = int(selected_channel_str.split(" ")[1])

# --- 参数设置 ---
st.sidebar.divider()
window_sec = st.sidebar.slider("窗口宽度 (秒)", 2, 8, 4)
y_range = st.sidebar.slider("Y轴范围 (uV)", 50, 500, 200)

# --- [新功能 3] 播放速度上限提高 ---
speed_step = st.sidebar.slider("播放步进 (秒/帧)", 0.1, 3.0, 0.1, help="数值越大，播放越快")

st.sidebar.divider()

# --- 播放控制 ---
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("▶️ 播放"):
    st.session_state.is_running = True
if col2.button("⏸️ 暂停"):
    st.session_state.is_running = False
if col3.button("🔄 重置"):
    st.session_state.is_running = False
    st.session_state.current_time = 0.0
    st.rerun()

st.sidebar.markdown(f"⏱️ **时间**: `{st.session_state.current_time:.2f} s`")

# ==========================================
# 4. 主界面：绘图逻辑
# ==========================================
st.title("🏥 胎儿心电实时提取系统 (Single Channel View)")

if st.session_state.stream is None:
    st.info("👈 请在左侧上传 TXT 文件以开始监测")
    st.stop()

# --- [新增] 指标显示区 ---
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    bpm_placeholder = st.empty()
with metric_col2:
    rr_placeholder = st.empty()
with metric_col3:
    status_placeholder = st.empty()

# 初始显示
bpm_placeholder.metric("❤️ Fetal Heart Rate", "-- BPM")
rr_placeholder.metric("📏 Mean RR Interval", "-- s")
status_placeholder.info("等待数据处理...")

# 图表占位符
chart_placeholder = st.empty()


def draw_plot(start_time):
    """
    绘制单帧图像：上下两张子图，并计算心率指标
    """
    stream = st.session_state.stream
    core = st.session_state.core

    # 获取所有通道的数据块
    raw_dict, duration = stream.get_data_chunk(start_time, window_sec)

    if raw_dict is None or duration < 0.1:
        return None  # 数据读完了

    # --- 获取选中通道的数据 ---
    all_channels = list(raw_dict.keys())
    if selected_ch_idx >= len(all_channels):
        st.error("选择的通道索引超出了文件实际通道数")
        return None

    target_col_name = all_channels[selected_ch_idx]
    raw_seg = raw_dict[target_col_name]

    # --- AI 推理与严格处理 ---
    try:
        raw_clean, fecg_pred = core.process_segment(raw_seg)

        # === [新增] 实时指标计算与更新 ===
        metrics = core.calculate_fhr_metrics(fecg_pred, fs=200)  # 假设 FECG 输出是 200Hz

        if metrics:
            bpm_val = f"{metrics['bpm']:.1f}"
            rr_val = f"{metrics['rr_mean']:.3f}"
            bpm_placeholder.metric("❤️ Fetal Heart Rate", f"{bpm_val} BPM")
            rr_placeholder.metric("📏 Mean RR Interval", f"{rr_val} s")
            status_placeholder.success("Signal Quality: Good")
        else:
            bpm_placeholder.metric("❤️ Fetal Heart Rate", "-- BPM")
            rr_placeholder.metric("📏 Mean RR Interval", "-- s")
            status_placeholder.warning("Signal Quality: Weak (No Peaks)")
        # ================================

    except Exception as e:
        return None

    # --- 绘图逻辑 ---
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)

    t_axis = np.linspace(start_time, start_time + duration, int(duration * 250))
    min_len = min(len(t_axis), len(raw_clean), len(fecg_pred))

    # 子图 1: 母体心电
    ax1.plot(t_axis[:min_len], raw_clean[:min_len], color='#2c3e50', lw=1.2)
    ax1.set_title(f"Maternal ECG (Processed) - {selected_channel_str}", fontsize=11, fontweight='bold', loc='left')
    ax1.set_ylabel("Amplitude (uV)", fontweight='bold', fontsize=9)
    ax1.set_ylim(-y_range, y_range)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(axis='y', labelsize=8)
    ax1.text(0.01, 0.85, "Processed Input", transform=ax1.transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 子图 2: 胎儿心电
    ax2.plot(t_axis[:min_len], fecg_pred[:min_len], color='#e74c3c', lw=1.2)
    ax2.set_title(f"Fetal ECG (Extracted) - {selected_channel_str}", fontsize=11, fontweight='bold', loc='left',
                  color='#c0392b')
    ax2.set_ylabel("Amplitude (uV)", fontweight='bold', fontsize=9)
    ax2.set_ylim(-y_range, y_range)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=8)
    ax2.text(0.01, 0.85, "DIFF-FECG Output", transform=ax2.transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), color='#c0392b')

    # 设置 X 轴
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_xlim(start_time, start_time + window_sec)

    plt.tight_layout(pad=1.2, h_pad=0.5)
    return fig


# ==========================================
# 5. 动画循环
# ==========================================
if st.session_state.is_running:
    while True:
        fig = draw_plot(st.session_state.current_time)

        if fig is None:
            st.session_state.is_running = False
            st.success("✅ 数据回放结束")
            break

        chart_placeholder.pyplot(fig)
        plt.close(fig)

        # 使用用户设定的“播放步进”来更新时间
        st.session_state.current_time += speed_step

        # 稍微休眠，给浏览器渲染留时间
        time.sleep(0.02)

else:
    # 暂停状态：只画单帧
    fig = draw_plot(st.session_state.current_time)
    if fig:
        chart_placeholder.pyplot(fig)
        plt.close(fig)