import streamlit as st  # 导入Streamlit用于构建网页界面
import numpy as np  # 数值计算库
import time  # 用于睡眠控制循环节奏
import pandas as pd  # 数据处理库（当前未直接使用）
import io
from data_stream import MockECGStreamer, NpyECGStreamer  # 引入模拟数据流类
from inference_core import InferenceEngine  # 引入推理引擎
from utils_vis import plot_ecg_interactive  # 引入绘图函数

# --- 页面配置 ---
st.set_page_config(page_title="AI Fetal Monitor (ADDB)", layout="wide", page_icon="👶")  # 设置页面标题布局和图标

# --- 侧边栏：控制面板 ---
st.sidebar.title("控制面板")  # 侧边栏标题
run_simulation = st.sidebar.toggle("开始实时监测", value=False)  # 开关控制是否启动实时监测

# 数据源选择：默认 ADDB，也可选择上传 NPY
data_source = st.sidebar.radio("数据来源", ["ADDB 演示", "NPY 文件"], index=0)

# ADDB 选项
patient_map = {
    "r01 (ADDB)": 0,
    "r04 (ADDB)": 1,
    "r07 (ADDB)": 2,
    "r08 (ADDB)": 3,
    "r10 (ADDB)": 4
}
selected_label = st.sidebar.selectbox("选择病人", list(patient_map.keys()), disabled=(data_source != "ADDB 演示"))
patient_idx = patient_map[selected_label]

# NPY 选项
npy_file = st.sidebar.file_uploader("上传 NPY 母体信号", type=["npy"], disabled=(data_source != "NPY 文件"))
npy_fs = st.sidebar.number_input("NPY 采样率 (Hz)", min_value=50, max_value=2000, value=200, step=50,
                                disabled=(data_source != "NPY 文件"))

# 模型选择 (保持原逻辑)
model_choice = st.sidebar.selectbox("选择模型", ["mkf2_improved", "own"])  # 模型选择下拉框

# --- 初始化状态 ---
if 'buffer_aecg' not in st.session_state:  # 如果尚未创建 AECG 缓冲
    st.session_state.buffer_aecg = np.zeros(1000)  # 5秒缓冲区（200Hz*5秒）
if 'buffer_fecg' not in st.session_state:  # 如果尚未创建 FECG 缓冲
    st.session_state.buffer_fecg = np.zeros(1000)  # 同步长度的FECG缓冲
if 'history_fhr' not in st.session_state:  # 如果尚未创建 FHR 历史
    st.session_state.history_fhr = []  # 存储历史心率
if 'stream_source' not in st.session_state:
    st.session_state.stream_source = 'addb'
if 'npy_channel' not in st.session_state:
    st.session_state.npy_channel = 0
if 'npy_channels' not in st.session_state:
    st.session_state.npy_channels = 1
if 'npy_file_bytes' not in st.session_state:
    st.session_state.npy_file_bytes = None


# --- 加载资源 ---
@st.cache_resource
def get_engine(name):  # 缓存创建推理引擎
    # 【修改点 3】强制指定 db='addb'
    return InferenceEngine(model_name=name, db='addb')  # 返回推理引擎实例


@st.cache_resource
def get_streamer(idx):
    return MockECGStreamer(db='addb', test_idx=idx)


engine = get_engine(model_choice)

# 根据数据源决定使用的 streamer
streamer = None
source_label = ""
if data_source == "ADDB 演示":
    streamer = get_streamer(patient_idx)
    if st.session_state.stream_source != 'addb':
        st.session_state.buffer_aecg = np.zeros(1000)
        st.session_state.history_fhr = []
    st.session_state.stream_source = 'addb'
    source_label = f"ADDB - {selected_label.split(' ')[0]}"
else:
    if npy_file is not None:
        # 缓存文件字节以便多次读取
        if st.session_state.npy_file_bytes is None or st.session_state.get('npy_filename') != npy_file.name:
            st.session_state.npy_file_bytes = npy_file.getvalue()
            st.session_state.npy_filename = npy_file.name
            st.session_state.npy_channel = 0
            # 探测通道数，逻辑与 NpyECGStreamer 保持一致
            arr = np.load(io.BytesIO(st.session_state.npy_file_bytes))
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            else:
                arr = arr
            st.session_state.npy_channels = arr.shape[0]
            if arr.ndim > 2:
                st.sidebar.error("仅支持 1D 或 2D npy 数据")
        # 通道选择
        channel_choices = list(range(st.session_state.npy_channels))
        selected_ch = st.sidebar.selectbox("选择通道", channel_choices, index=st.session_state.npy_channel,
                                           disabled=False)
        if selected_ch != st.session_state.npy_channel or st.session_state.stream_source != 'npy':
            st.session_state.npy_channel = selected_ch
            st.session_state.buffer_aecg = np.zeros(1000)
            st.session_state.history_fhr = []
        try:
            streamer = NpyECGStreamer(io.BytesIO(st.session_state.npy_file_bytes), channel=st.session_state.npy_channel,
                                      fs=int(npy_fs))
            st.session_state.stream_source = 'npy'
            source_label = f"NPY - {npy_file.name} - Ch{st.session_state.npy_channel}"
        except Exception as e:
            st.sidebar.error(f"NPY 加载失败: {e}")
            streamer = None
    else:
        st.sidebar.info("请上传 NPY 文件")

# --- 主界面布局 ---
st.title("👶 智能胎儿心电实时监测系统 (ADDB版)")  # 主标题

col1, col2, col3, col4 = st.columns(4)  # 创建四列用于指标显示
metric_fhr = col1.empty()  # 占位：胎心率
metric_rr = col2.empty()  # 占位：RR 间隔
metric_status = col3.empty()  # 占位：状态
metric_snr = col4.empty()  # 占位：数据源信息

st.subheader("实时波形 (Real-time Waveforms)")  # 子标题：波形
chart_aecg = st.empty()  # 占位：原始 AECG 图
chart_fecg = st.empty()  # 占位：重建 FECG 图

st.subheader("心率趋势 (FHR Trend)")  # 子标题：趋势
chart_trend = st.empty()  # 占位：心率趋势折线

# --- 实时循环 ---
if streamer is None:
    st.info("请选择数据源并完成加载后再启动")
elif run_simulation:  # 如果开启监测
    # 每次读取的点数（0.2秒），数据采样率为200Hz，次读取的数据点数，对应0.2秒的时间长度。
    # 在这个项目中，数据采样率为200Hz（每秒200个数据点），因此0.2秒的数据量为200 * 0.2 = 40个点
    chunk_size = 40
    #即每次循环迭代后暂停0.1秒。这个休眠用于控制实时监测循环的节奏，避免CPU过度占用，
    # 同时模拟数据的实时流速，确保界面更新频率合适（大约每0.1秒更新一次）
    sleep_time = 0.01  # 循环休眠时间

    while True:  # 主循环
        new_aecg, new_truth = streamer.get_next_chunk(chunk_size)  # 获取新的 AECG 片段及真值

        st.session_state.buffer_aecg = np.roll(st.session_state.buffer_aecg, -chunk_size)  # 缓冲左移
        st.session_state.buffer_aecg[-chunk_size:] = new_aecg  # 追加新片段

        window_data = st.session_state.buffer_aecg  # 当前窗口数据
        rec_fecg, peaks = engine.process_window(window_data)  # 推理得到 FECG 与峰位

        valid_peaks = peaks[peaks > 800]  # 过滤窗口中后半段的峰（可视化用）
        fhr, rr = engine.calculate_metrics(peaks)  # 计算心率与 RR

        # 更新 UI
        metric_fhr.metric("胎心率 (FHR)", f"{fhr:.0f} bpm", delta=f"{fhr - 140:.0f}")  # 显示心率及相对 140 的差
        metric_rr.metric("RR 间隔", f"{rr:.0f} ms")  # 显示 RR 间隔

        if fhr < 110 or fhr > 160:  # 判断心率是否异常
            metric_status.error("⚠️ 异常")  # 异常提示
        else:
            metric_status.success("✅ 正常")  # 正常提示

        metric_snr.info(f"数据源: {source_label}")  # 显示数据源信息

        # 使用 Plotly 画图

        # 1. 准备图表对象
        fig_aecg = plot_ecg_interactive(  # 绘制 AECG 波形
            window_data[-1000:],  # 最近 5 秒数据
            title="原始腹部信号 (Raw AECG)",  # 图标题
            color='#1f77b4'  # 线条颜色
        )


        # 2. 直接在占位符上绘制 (原地更新，不仅不报错，也不会乱跳)
        chart_aecg.plotly_chart(fig_aecg, use_container_width=True)  # 更新 AECG 图

        # 3. 同理处理 FECG
        display_len = 1000  # 显示长度 5 秒
        display_signal = rec_fecg[-display_len:]  # 取末尾信号
        start_idx = len(rec_fecg) - display_len  # 起始索引
        valid_peaks_vis = peaks[peaks >= start_idx] - start_idx  # 对可视峰值重定位

        fig_fecg = plot_ecg_interactive(  # 绘制重建 FECG 波形
            display_signal,  # 信号数据
            peaks=valid_peaks_vis,  # 峰值位置
            title="重建胎儿信号 (Reconstructed FECG)",  # 图标题
            color='#2ca02c'  # 线条颜色
        )

        # 直接绘制，不要加 key
        chart_fecg.plotly_chart(fig_fecg, use_container_width=True)  # 更新 FECG 图

        if fhr > 0:  # 若心率有效
            st.session_state.history_fhr.append(fhr)  # 追加历史
            if len(st.session_state.history_fhr) > 100:  # 限制长度
                st.session_state.history_fhr.pop(0)  # 移除最早数据
            # line_chart 不需要 key，它自动处理得很好
            chart_trend.line_chart(st.session_state.history_fhr, height=200)  # 更新趋势图

        time.sleep(sleep_time)  # 控制循环节奏

else:
    st.info("请点击左侧 '开始实时监测' 启动系统")  # 提示用户启动监测
