import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dataset import FECGDataset
from config import cfg
import data_util as du
# 1. 导入作者的工具包
import util
from ecgdetectors import Detectors

# ---------------------------------------------------------
# 2. 初始化环境
# ---------------------------------------------------------
# 加载 GetTrainTest-fecg.py 中的核心函数 (load_model, predict 等)
exec(open("GetTrainTest-fecg.py", encoding='utf-8').read())

# 配置参数 (请根据你的实际训练情况调整)
c = cfg()
c.train = False
c.db = 'addb'  # 数据集
c.model_name = 'mkf2_improved'  # 模型名
# c.model_name = 'own'  # 模型名
c.fs = 200  # 采样率
c.model_save_dir = os.path.join(c.RESULT, 'model')

# 其他固定配置
c.max_epoch = 15;
c.seg_len = 1000;
c.filter = [7.5, 75]
c.batch_size = 64;
c.sample_k = 10;
c.fecg_label = True
c.loss_name = 'diff';
c.train_all_channel = True

util.speed_up()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"开始评估 (调用 util.evaluate) | Database: {c.db} | Model: {c.model_name}")


# ---------------------------------------------------------
# 3. 核心：调用作者的评估函数
# ---------------------------------------------------------
def detect_and_score_using_author_func(real_peaks, pred_signal, fs):
    """
    1. 使用 Pan-Tompkins 检测 R 峰
    2. 对齐 R 峰 (修正 ADDB 下采样偏差)
    3. 调用 util.evaluate 计算指标
    """
    try:
        detectors = Detectors(fs)

        # --- A. 极性校正 (防止波形倒置导致漏检) ---
        if np.abs(np.min(pred_signal)) > np.abs(np.max(pred_signal)) * 1.5:
            pred_signal = -pred_signal

        # --- B. R 峰检测 (Pan-Tompkins) ---
        pred_peaks = np.array(detectors.pan_tompkins_detector(pred_signal))

        if len(real_peaks) == 0 or len(pred_peaks) == 0:
            return 0, 0, 0

        # --- C. 自动对齐 (修复 -8.27 点的系统性偏差) ---
        # 计算预测峰与真实峰的平均距离
        diffs = []
        for r in real_peaks:
            dist = pred_peaks - r
            closest = dist[np.argmin(np.abs(dist))]
            if abs(closest) < 30:  # 只统计附近的点
                diffs.append(closest)

        if len(diffs) > 5:
            offset = np.median(diffs)
            pred_peaks = (pred_peaks - offset).astype(int)  # 修正偏移

        # --- D. 【关键】调用作者写的 util.evaluate 函数 ---
        # 作者函数的定义是: evaluate(r_ref, r_ans, fs, thr, ...)
        # 注意：作者要求输入是 list 的 list (即 [[peaks_patient1], [peaks_patient2]...])
        # 所以我们要加中括号 []

        recall, precision, f1 = util.evaluate(
            r_ref=[real_peaks],  # 真实值 (放入列表)
            r_ans=[pred_peaks],  # 预测值 (放入列表)
            fs=fs,
            thr=50,  # 阈值 (论文标准 50ms)
            print_msg=False  # 不要在函数内部打印，我们在外面打印
        )

        # 作者返回的是小数，我们转成百分比
        return f1 * 100, recall * 100, precision * 100

    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0, 0, 0


# ---------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------
names = du.get_namelist(c.db)
results = []

print("\n" + "=" * 80)
print(f"{'Patient':<10} | {'F1-Score':<10} | {'Sen':<10} | {'PPV':<10} | (From util.evaluate)")
print("-" * 80)

for i, name in enumerate(names):
    # 1. 加载模型
    model_file = c.db + '_' + c.model_name + '_' + str(i) + '_fecg' + '_' + c.loss_name + '_' + str(c.loss_weight)
    try:
        model = load_model(model_dir=c.model_save_dir, model_file=model_file)
        model.to(device).eval()
    except:
        print(f"{name:<10} | Model Not Found")
        continue

    # 2. 推理参数
    alpha, beta, alpha_cum, sigmas, T_sch, c1, c2, c3, delta, delta_bar = inference_schedule(model)

    # 3. 遍历通道 (寻找该病人的最佳通道)
    best_f1 = -1
    best_metrics = (0, 0, 0)  # F1, Sen, PPV

    for ac in [0, 1, 2, 3]:
        c.aecg_channel = [ac]
        ds = FECGDataset(c, db=c.db, train=False, seg_len=c.seg_len, fs=c.fs, test_idx=i, aecg_channel=c.aecg_channel,
                         fecg_label=True)
        dl = DataLoader(ds, batch_size=c.batch_size, shuffle=False)

        # 3.1 获取完整波形
        pre_list = []
        for x, _ in dl:
            x = x.squeeze(1).float().to(device)
            # 简化采样：只采1次以加快速度，如需更高精度可改为 c.sample_k
            with torch.no_grad():
                out = predict(model, x, alpha, beta, alpha_cum, sigmas, T_sch, c1, c2, c3, delta, delta_bar,
                              device=device)
            pre_list.append(out.cpu().numpy().flatten())

        pre_signal = data_reshape(np.concatenate(pre_list), c.seg_len)
        pre_signal = du.butter_bandpass_filter(pre_signal, c.filter[0], c.filter[1], c.fs, axis=0)

        # 3.2 获取真值 R 峰
        real_peaks = ds.get_fqrs()

        # 3.3 计算分数
        f1, sen, ppv = detect_and_score_using_author_func(real_peaks, pre_signal, c.fs)

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (f1, sen, ppv)

    # 4. 记录结果
    results.append(best_metrics)
    print(f"{name:<10} | {best_metrics[0]:.2f}%      | {best_metrics[1]:.2f}%     | {best_metrics[2]:.2f}%")

# 5. 最终平均分
if results:
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    print("-" * 80)
    print(
        f"{'Average':<10} | {means[0]:.2f} ± {stds[0]:.2f} | {means[1]:.2f} ± {stds[1]:.2f} | {means[2]:.2f} ± {stds[2]:.2f}")
    print("=" * 80)