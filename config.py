from yacs.config import CfgNode as CN
import numpy as np
import os
config = CN()
###数据库###
# 数据集	    | 类型	   |  数据量	  |        论文核心用途	            |      代码地位
# ADDB	    | 真实	   | 小 (5例)  | 选演示，可视化效果对比，基准测试	    | 默认配置，新手复现首选
# BDDB	    | 真实	   | 中 (12例) | 综合验证，抗噪实验(加噪音测试)	    | 进阶训练，验证模型稳定性
# FECGSYNDB	| 合成	   | 大/多场景  | 泛化测试，测试特定极端条件(如低信噪比) |	补充测试，验证鲁棒性
# ================= 核心修改开始 =================
# ================= 智能路径修复开始 =================
# 1. 获取当前 config.py 所在的目录 (即 DIFF-FECG-main)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 定义两种可能的 data 路径
# 可能性 A: 在项目里面 (DIFF-FECG-main/data)
path_inner = os.path.join(current_dir, 'data')
# 可能性 B: 在项目旁边 (Desktop/data)
path_outer = os.path.join(os.path.dirname(current_dir), 'data')

# 3. 自动探测：哪里有 'adfecgdb' 就用哪里
if os.path.exists(os.path.join(path_inner, 'adfecgdb')):
    print(f"✅ 在项目内部找到数据: {path_inner}")
    data_root = path_inner
elif os.path.exists(os.path.join(path_outer, 'adfecgdb')):
    print(f"✅ 在项目外部找到数据: {path_outer}")
    data_root = path_outer
else:
    # 两个都找不到，报错提示
    print(f"❌ 警告：无法在 {path_inner} 或 {path_outer} 找到 adfecgdb 数据集！")
    print("请检查文件夹名是否拼写正确 (必须是 adfecgdb 不是 addb)")
    # 默认回退到内部路径，防止变量未定义
    data_root = path_inner

# 4. 设置最终路径
config.ADFECGDB = os.path.join(data_root, 'adfecgdb')
config.BDDB = os.path.join(data_root, 'bddb')
config.FECGSYNDB = os.path.join(data_root, 'fecgsyndb')

# 结果保存路径 (保持在项目内)
config.RESULT = os.path.join(current_dir, 'resource')
# ================= 智能路径修复结束 =================


###实验用的数据库
config.db = 'addb'

config.fecg_label = True
config.mecg_label = False


##preprocess
config.aecg_channel = [0]  ##使用的通道，总共0,1,2,3。[0,1,2,3]表示使用4个通道
config.smooth = False ##数据是否平滑，没啥用，可以忽略
config.seg_len = 992 ##输入模型的长度
config.fs = 200 ###输入数据的采样频率，这个不是数据库的原始采样频率，设定好后，数据会重采样成这个频率
config.smooth_label = False ###设为false，暂时不用
config.filter = [7.5, 75]#[3,90]#7.5,75

##model
config.train = True 
config.model_name = 'fecg' ##暂时只是保存模型数据的文件夹名
config.batch_size = 32 
config.max_epoch = 80 ###训练次数
config.loss_name = 'diff'
config.ckpt = 'ckpt' 
config.lr =0.0001#0.0001
config.lr_policy = 'plateau' ###这个还没用上
config.train_all_channel = True 

config.loss_weight = 0.5
config.sample_k = 1

##evaluate
config.seed = 1
config.qrs_thr = 50 
config.ignore_sec = 0
config.test_idx = 0



##code test
config.show_pic = False
config.show_bar = False
 

def cfg():
    return config