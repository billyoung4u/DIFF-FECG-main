import sys  # 系统路径操作
import os  # 路径处理
import torch  # 深度学习框架
import numpy as np  # 数值计算
import streamlit as st  # Web 界面
from scipy.signal import find_peaks  # 峰值检测

# 引入项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 加到路径
import data_util as du  # 数据处理工具
import util  # 常用工具

# 动态加载 GetTrainTest 中的函数
exec(open(os.path.join(os.path.dirname(__file__), '../GetTrainTest-fecg.py'), encoding='utf-8').read())  # 动态执行脚本


class InferenceEngine:
    def __init__(self, model_name, db='addb'):  # 初始化推理引擎
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
        self.model, self.params = self._load_model_cached(model_name, db)  # 加载模型和参数

    @st.cache_resource  # Streamlit 缓存装饰器，保证模型只加载一次
    def _load_model_cached(_self, model_name, db):  # 内部缓存模型加载
        from config import cfg  # 延迟导入配置
        c = cfg()  # 创建配置
        # 构造路径 (请根据你实际的 best model 修改 test_idx 等)
        # 这里默认加载第0个病人的模型作为演示通用模型
        # 实际部署建议用混合训练的模型
        model_file = f"{db}_{model_name}_0_fecg_diff_0.5"  # 模型文件名
        save_dir = os.path.join(c.RESULT, 'model')  # 保存路径

        print(f"Loading Model from: {save_dir}/{model_file}")  # 日志
        try:
            model = load_model(model_dir=save_dir, model_file=model_file)  # 加载模型
            model.to(_self.device).eval()  # 移到设备并评估模式

            # 获取扩散参数
            params = inference_schedule(model)  # 获取扩散推理参数
            return model, params  # 返回模型与参数
        except Exception as e:
            st.error(f"模型加载失败: {e}")  # 显示错误
            return None, None  # 失败返回空

    def process_window(self, aecg_signal):
        """
        输入一段 AECG (如 5秒 1000点)，输出重建的 FECG 和 R峰
        """
        if self.model is None: return np.zeros_like(aecg_signal), []  # 无模型则返回空

        # 1. 预处理 (归一化 + 维度调整)
        # 修正：模型内部会自动 unsqueeze，这里只需要变成 [Batch, Length] 即可
        x = torch.from_numpy(aecg_signal).float().unsqueeze(0).to(self.device)  # 转张量并上设备

        # 2. 推理 (为了实时性，sample_k 设为 1)
        alpha, beta, alpha_cum, sigmas, T_sch, c1, c2, c3, delta, delta_bar = self.params  # 解包参数
        with torch.no_grad():  # 关闭梯度
            out = predict(self.model, x, alpha, beta, alpha_cum, sigmas, T_sch, c1, c2, c3, delta, delta_bar,
                          device=self.device)  # 调用推理

        rec_fecg = out.cpu().numpy().flatten()  # 转回 numpy 并压平

        # 3. 滤波 (可选，去毛刺)
        rec_fecg = du.butter_bandpass_filter(rec_fecg.reshape(1, -1), 7.5, 75, 200, axis=1).flatten()  # 带通滤波

        # 4. R 峰检测
        # 极性校正
        if np.abs(np.min(rec_fecg)) > np.abs(np.max(rec_fecg)) * 1.5:  # 若负峰过大
            rec_fecg = -rec_fecg  # 取反

        peaks, _ = find_peaks(rec_fecg, distance=200 * 0.3, height=0.5)  # 峰值检测

        return rec_fecg, peaks  # 返回重建信号和峰

    def calculate_metrics(self, peaks, fs=200):  # 计算心率和 RR
        if len(peaks) < 2:  # 峰不足
            return 0, 0  # 返回零

        # 计算 RR 间隔 (秒)
        rr_intervals = np.diff(peaks) / fs  # 差分后换算秒
        mean_rr = np.mean(rr_intervals)  # 平均 RR

        # 计算 FHR (bpm)
        current_fhr = 60 / mean_rr if mean_rr > 0 else 0  # 心率换算

        return current_fhr, mean_rr * 1000  # ms 返回
