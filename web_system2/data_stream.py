import pandas as pd
import numpy as np


class TxtDataStream:
    def __init__(self, uploaded_file):
        self.df = self._parse_txt(uploaded_file)
        # 自动识别前 6 个 EXG 通道
        self.channels = [c for c in self.df.columns if 'EXG Channel' in c]
        if not self.channels:
            self.channels = self.df.columns[1:9] if self.df.shape[1] >= 9 else []
        self.channels = self.channels[:6]  # 限制前6个

        self.fs = 250
        self.total_len = len(self.df)
        self.cursor = 0  # 当前读取指针

    def _parse_txt(self, file_obj):
        # Streamlit 的 file_obj 需要重新定位指针
        file_obj.seek(0)
        df = pd.read_csv(file_obj, comment='%', header=0, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        return df

    def get_total_duration(self):
        return self.total_len / self.fs

    def get_data_chunk(self, start_sec, duration_sec):
        """获取指定时间段的数据"""
        start_idx = int(start_sec * self.fs)
        end_idx = int((start_sec + duration_sec) * self.fs)

        # 边界检查
        if start_idx >= self.total_len:
            return None

        # 数据截取
        chunk_data = {}
        real_end = min(end_idx, self.total_len)

        for ch in self.channels:
            chunk_data[ch] = self.df[ch].values[start_idx:real_end]

        # 如果不够长（尾部），补零以保持长度一致（可选，这里为了显示连续性直接返回短的也行）
        return chunk_data, (real_end - start_idx) / self.fs