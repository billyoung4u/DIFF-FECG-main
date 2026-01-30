import plotly.graph_objects as go  # 导入 Plotly 绘图对象
import numpy as np  # 数值计算
import streamlit as st  # Streamlit UI


def plot_ecg_interactive(signal, peaks=None, title="ECG Signal", color='blue', height=250):
    """
    使用 Plotly 绘制交互式 ECG 波形，并标记 R 峰
    """
    # 创建画布
    fig = go.Figure()  # 初始化图表对象

    # 1. 绘制波形线
    time_axis = np.arange(len(signal)) / 200.0  # 假设 200Hz，转换为秒
    fig.add_trace(go.Scatter(
        x=time_axis,  # 时间轴
        y=signal,  # 信号值
        mode='lines',  # 折线模式
        name='Signal',  # 图例名
        line=dict(color=color, width=1.5)  # 线条样式
    ))

    # 2. 标记 R 峰 (如果有)
    if peaks is not None and len(peaks) > 0:  # 若传入峰值
        # 过滤掉超出当前显示范围的峰 (虽然 Plotly 会自动处理，但为了性能)
        valid_peaks = [p for p in peaks if p < len(signal)]  # 保留范围内的峰
        peak_times = np.array(valid_peaks) / 200.0  # 峰对应的时间
        peak_values = signal[valid_peaks]  # 峰对应的幅值

        fig.add_trace(go.Scatter(
            x=peak_times,  # 峰时间
            y=peak_values,  # 峰幅值
            mode='markers',  # 点模式
            name='R-Peak',  # 图例名
            marker=dict(color='red', size=8, symbol='x')  # 标记样式
        ))

    # 3. 美化布局
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),  # 标题设置
        height=height,  # 高度
        margin=dict(l=10, r=10, t=30, b=10),  # 减少留白
        xaxis=dict(
            title="Time (s)",  # 横轴标题
            showgrid=True,  # 显示网格
            gridcolor='rgba(200,200,200,0.2)'  # 网格颜色
        ),
        yaxis=dict(
            showgrid=True,  # 显示网格
            gridcolor='rgba(200,200,200,0.2)',  # 网格颜色
            zeroline=False  # 不显示零线
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        paper_bgcolor='rgba(0,0,0,0)',  # 透明纸张
        showlegend=True  # 显示图例
    )

    return fig  # 返回图表对象
