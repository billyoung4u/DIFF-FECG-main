from argparse import ArgumentParser
# 2. 导入 PyTorch 相关的并行计算工具（用于多显卡训练，但本项目主要是单卡）
from torch.cuda import device_count
from torch.multiprocessing import spawn

# 3. 导入数值计算库，处理数组和矩阵
import numpy as np
import os# 4. 导入操作系统接口，用于路径管理、文件读写
import torch# 5. 导入 PyTorch 深度学习框架核心
import torch.nn as nn# 6. 导入 PyTorch 的神经网络模块（如卷积层、全连接层）
from torch.nn.parallel import DistributedDataParallel # 7. 导入分布式数据并行模块（用于多卡训练）
from torch.utils.tensorboard import SummaryWriter# 8. 导入 TensorBoard 工具，用于记录训练过程的 Loss 曲线（可视化）
from tqdm import tqdm# 9. 导入进度条库，用于在控制台显示训练进度

from torch.utils.data import DataLoader# 10. 导入 PyTorch 数据加载器，负责把数据分批次喂给模型
from dataset import FECGDataset# 11. 从 dataset.py 文件导入自定义的数据集类 FECGDataset
from config import cfg# 12. 从 config.py 文件导入配置加载函数 cfg

import random# 13. 导入随机数库，用于设置随机种子
import matplotlib.pyplot as plt# 14. 导入绘图库，用于画波形图
from sympy import symbols, Eq, solve# 15. 导入符号计算库（这里可能是冗余导入，主要在数据处理中用到数学公式求解）

import data_util as du# 16. 从 data_util.py 导入自定义的数据工具函数（如读取 EDF 文件）
import util# 17. 从 util.py 导入自定义的通用工具函数（如计算评估指标）

# 这一行的意思是：打开当前目录下的 "GetTrainTest-fecg.py" 文件，
# 读取它的全部内容，并用 exec() 函数在当前位置立即执行它。
# 效果等同于：把那个文件里的所有代码复制粘贴到了这里。
# 定义了 train()（训练函数）、test()（测试函数） 和 DiffLearner（模型包装类）。
# 运行这一行后，你的程序内存里就有了这些函数，下面才能调用 train()。
exec(open("GetTrainTest-fecg.py", encoding='utf-8').read())
c = cfg()# 初始化配置对象，从 config.py 获取默认参数


##数据集r01有30*64条数据，每64条为一个batch，则一个epoch有30个batch
##tran中硬编码max_step=200*40，每个step对应一个batch，也就是让每个数据集训练8000/30=266.6个epoch
c.max_epoch =15# 修改最大训练轮数为15,此处没用，因为train中硬编码了epoch

seed = 2010#定义随机种子（虽然定义了，但要看后面有没有调用 util.setup_seed(seed) 来生效）

c.db = 'addb'#  指定使用的数据库名称，'addb' 是腹部胎儿心电数据库
test_idx = 0# 5. 初始化测试对象的索引（后面循环会覆盖它，这里只是占位）
###
c.train = True# 6. 【重要开关】设置为 True 表示进行训练；False 表示直接加载模型进行测试
# c.model_name = 'own'# 7. 给模型起个名字，用于保存文件时区分
c.model_name ='mkf2_improved'
c.seg_len = 1000# 8. 设置输入数据的切片长度，1000 个采样点（5秒 x 200Hz）
c.filter = [7.5,75]# 9. 设置带通滤波器的范围：7.5Hz 到 75Hz，用于去除噪声
c.batch_size = 64# 10. 设置批次大小（Batch Size），一次训练喂给模型 64 条数据

c.loss_weight = 0.5# 11. 损失函数的权重，用于平衡不同 Loss 之间的关系

c.sample_k = 10# 12. 扩散模型采样次数（k=10 表示重复采样 10 次取平均，提高质量但变慢）

c.fecg_label = True# 13. 是否使用胎儿心电（FECG）作为标签
c.aecg_channel = [1]# 14. 设置使用的通道，[1] 表示只用第 2 个通道（索引从0开始）
c.loss_name = 'diff'# 15. 损失函数名称，这里使用 'diff'（可能是差分损失或扩散相关损失）
c.model_save_dir = c.RESULT + '/model'# 16. 设置模型保存的文件夹路径，c.RESULT 在 config.py 里定义



c.train_all_channel = True# 17. 训练时是否使用所有通道（True 表示把所有腹部通道的数据都拿来训练）
contain_label = True# 18. 数据是否包含标签（用于数据加载逻辑）


c.show_pic = True# 19. 是否显示波形图片（False 关闭，防止弹出大量窗口）
c.show_bar  = True# 20. 【重要】是否显示进度条（False 关闭，True 开启）
util.speed_up()# 1. 调用 util 中的加速函数（设置 cudnn.benchmark=True，让 GPU 跑得更快）
data_list = du.get_namelist(c.db)# 2. 获取数据库中所有文件的列表（例如 ['r01.edf', 'r02.edf', ...]）
pccs = []# 保存皮尔逊相关系数
msg = []# 保存日志信息




# 1. 初始化一个列表，用来保存所有病人的最佳结果
all_metrics = []

# 4. 【大循环开始】遍历每一个病人，轮流做“测试集”
# 例如：第 1 次循环，用 r01 做测试，其他人做训练；第 2 次用 r02 做测试...
# --- 替换 main.ipynb 中的主循环部分 ---

for test_idx in range(len(data_list)):
    print('-' * 50)
    print(f'Processing Patient {test_idx} ({data_list[test_idx]})...')

    # 构造模型文件名
    model_file = c.db + '_' + c.model_name + '_' + str(test_idx) + '_fecg' + '_' + c.loss_name + '_' + str(
        c.loss_weight)

    # --- 训练阶段 ---
    if c.train:
        print('Training started...')
        c.aecg_channel = [0, 1, 2, 3]
        # 调用 GetTrainTest-fecg.py 里的 train 函数
        # 注意：train() 函数内部会自动保存模型到 c.model_save_dir
        train()

    # --- 测试阶段 ---
    print('Testing started...')
    max_score = -99999
    best_res = None

    # 遍历 4 个通道，寻找最佳通道
    for aecg_channel in [0, 1, 2, 3]:
        c.aecg_channel = [aecg_channel]
        try:
            # 调用 test 函数获取 6 个指标
            # pcc, snr, prd, spec_corr, mae, mse
            curr_res = test(c.model_save_dir)

            # 这里以 PCC 为主导打分，也可以改成 pcc + 0.01*snr
            curr_pcc = curr_res[0]

            if curr_pcc > max_score:
                max_score = curr_pcc
                best_res = curr_res
        except Exception as e:
            print(f"Channel {aecg_channel} error: {e}")
            continue

    # --- 保存并打印当前病人的结果 ---
    if best_res is not None:
        all_metrics.append(best_res)  # 存入总账本
        pcc, snr, prd, spec_corr, mae, mse = best_res
        print(f'>>> Patient {test_idx} Best Result:')
        print(f'    PCC: {pcc:.4f} | SNR: {snr:.4f} | PRD: {prd:.4f} | MAE: {mae:.4f}')
    else:
        print(f'>>> Patient {test_idx}: No valid results.')

# --- 循环结束，开始计算平均分 ---
print('=' * 50)
print('FINAL AVERAGE RESULTS (Compare with Paper Table I)')
print('=' * 50)

if len(all_metrics) > 0:
    # 将列表转换为 numpy 数组，方便计算 [N_patients, 6_metrics]
    results_arr = np.array(all_metrics)

    # 计算均值和标准差
    means = np.mean(results_arr, axis=0)
    stds = np.std(results_arr, axis=0)

    # 指标顺序对应: pcc, snr, prd, spec_corr, mae, mse
    metric_names = ['PCC', 'SNR', 'PRD', 'Spec_Corr', 'MAE', 'MSE']

    for i, name in enumerate(metric_names):
        print(f"{name:10s}: {means[i]:.4f} ± {stds[i]:.4f}")

    print('-' * 50)
else:
    print("No metrics collected.")

