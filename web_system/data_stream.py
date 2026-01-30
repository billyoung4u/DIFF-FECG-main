import sys  # 系统路径操作
import os  # 文件与路径
import numpy as np  # 数值计算
import time  # 时间控制（此处未使用）

# 将上级目录加入路径，以便导入原项目的包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 增加项目根目录到路径
from dataset import FECGDataset  # 导入数据集类
from config import cfg  # 导入配置


class MockECGStreamer:
    def __init__(self, db='addb', test_idx=0, channel=0, fs=200):  # 初始化流对象
        # 初始化配置
        self.c = cfg()  # 创建配置实例
        self.c.db = db  # 指定数据库
        self.c.fs = fs  # 采样率，此处意味着每秒钟从心电信号中采集 200 个数据点
        self.c.train = False  # 推理模式
        self.c.seg_len = 1000  # 片段长度

        # 加载一个完整病人的数据
        print(f"正在加载 {db} 数据库，病人索引 {test_idx} ...")  # 日志
        # 确保 fecg_label=True 以便我们获取真值用于对比
        dataset = FECGDataset(self.c, db=db, train=False, seg_len=1000, test_idx=test_idx,
                              aecg_channel=[channel], fecg_label=True)  # 创建数据集

        # 获取完整的原始信号 (这里利用 dataset 的逻辑拼凑或者直接读取)
        # 为简单起见，我们直接从 dataset 中提取所有片段拼接
        self.full_aecg = []  # 存储拼接的 AECG
        self.full_fecg = []  # 存储拼接的 FECG 真值

        for i in range(len(dataset)):  # 遍历所有片段
            noisy, clean = dataset[i]  # 获取带噪与真值
            self.full_aecg.append(noisy.flatten())  # 拼接 AECG
            self.full_fecg.append(clean.flatten())  # 拼接 FECG

        if len(self.full_aecg) > 0:  # 若存在数据
            self.full_aecg = np.concatenate(self.full_aecg)  # 合并为长序列
            self.full_fecg = np.concatenate(self.full_fecg)  # 合并真值
        else:
            # 防止空数据报错
            self.full_aecg = np.zeros(1000)  # 备用空数据
            self.full_fecg = np.zeros(1000)  # 备用空真值

        self.current_ptr = 0  # 当前指针
        self.total_len = len(self.full_aecg)  # 总长度

    def get_next_chunk(self, chunk_size=200):
        """
        模拟获取下一个数据块（例如每次取 1秒=200点）
        """
        if self.current_ptr + chunk_size >= self.total_len:  # 若到末尾
            self.current_ptr = 0  # 循环播放

        chunk_aecg = self.full_aecg[self.current_ptr: self.current_ptr + chunk_size]  # 当前 AECG 片段
        chunk_fecg = self.full_fecg[self.current_ptr: self.current_ptr + chunk_size]  # 当前 FECG 真值

        self.current_ptr += chunk_size  # 指针前移
        return chunk_aecg, chunk_fecg  # 返回片段

