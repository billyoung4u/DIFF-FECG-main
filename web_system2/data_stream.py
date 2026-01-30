import pandas as pd
import numpy as np
import pyedflib
import tempfile
import os


class TxtDataStream:
    """
    处理 OpenBCI 导出的 TXT 文件
    """

    def __init__(self, uploaded_file):
        self.df = self._parse_txt(uploaded_file)
        # 自动识别前 6 个 EXG 通道
        self.channels = [c for c in self.df.columns if 'EXG Channel' in c]
        if not self.channels:
            self.channels = self.df.columns[1:9] if self.df.shape[1] >= 9 else []
        self.channels = self.channels[:6]  # 限制前6个

        self.fs = 250  # OpenBCI 默认采样率
        self.total_len = len(self.df)

    def _parse_txt(self, file_obj):
        file_obj.seek(0)
        df = pd.read_csv(file_obj, comment='%', header=0, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        return df

    def get_data_chunk(self, start_sec, duration_sec):
        """获取指定时间段的数据"""
        start_idx = int(start_sec * self.fs)
        end_idx = int((start_sec + duration_sec) * self.fs)

        if start_idx >= self.total_len:
            return None, 0

        chunk_data = {}
        real_end = min(end_idx, self.total_len)

        # 统一将 key 映射为 "Channel 0", "Channel 1"... 方便前端选择
        for i, ch in enumerate(self.channels):
            col_name = f"Channel {i}"
            chunk_data[col_name] = self.df[ch].values[start_idx:real_end]

        return chunk_data, (real_end - start_idx) / self.fs


class EdfFileStreamer:
    """
    处理标准 EDF 格式文件
    """

    def __init__(self, uploaded_file):
        # 1. 保存为临时文件 (pyedflib 需要物理路径)
        suffix = ".edf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_file.getvalue())
            self.temp_path = tfile.name

        self.data = {}
        self.fs = 250  # 默认采样率，后面会从文件读取更新
        self.total_len = 0

        try:
            self._load_edf()
        finally:
            # 读取完内存后删除临时文件
            if os.path.exists(self.temp_path):
                try:
                    os.remove(self.temp_path)
                except:
                    pass

    def _load_edf(self):
        f = pyedflib.EdfReader(self.temp_path)

        # 获取采样率 (假设所有通道一致，取第0个)
        try:
            self.fs = f.getSampleFrequency(0)
        except:
            self.fs = 250  # 兜底

        n_channels = f.signals_in_file
        # 获取总点数
        self.total_len = f.getNSamples()[0]

        # 读取前 6 个通道 (或所有通道)
        limit = min(n_channels, 6)
        for i in range(limit):
            sig = f.readSignal(i)
            # 统一命名为 "Channel X"
            self.data[f"Channel {i}"] = sig

        # 如果通道不足6个，补全(可选，为了不报错)
        for i in range(limit, 6):
            self.data[f"Channel {i}"] = np.zeros(self.total_len)

        f.close()

    def get_data_chunk(self, start_sec, duration_sec):
        """
        接口与 TxtDataStream 保持完全一致
        """
        start_idx = int(start_sec * self.fs)
        end_idx = int((start_sec + duration_sec) * self.fs)

        if start_idx >= self.total_len:
            return None, 0

        chunk_data = {}
        real_end = min(end_idx, self.total_len)

        # 遍历所有已加载的通道
        for col_name, full_signal in self.data.items():
            chunk_data[col_name] = full_signal[start_idx:real_end]

        return chunk_data, (real_end - start_idx) / self.fs