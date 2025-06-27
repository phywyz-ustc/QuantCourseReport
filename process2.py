import os
import pandas as pd
import numpy as np
from collections import defaultdict

# 路径设置
data_dir = r"D:\AI4Charting\大作业\after"
output_dir = r"D:\AI4Charting\大作业\processed"
os.makedirs(output_dir, exist_ok=True)

# 初始化
data = {}
labels = {}
data_type = {}
windows_by_date = defaultdict(list)
num_data = 0

# 遍历 CSV 文件
for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'):
        continue
    print(f"Processing {filename}...")
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    if len(df) < 13:
        continue

    # 提取基础特征
    features = df[["open", "high", "low", "close", "volume"]].values
    dates = df["date"].values

    N = len(df)
    print(f"Number of data points in {filename}: {N}")
    for i in range(N - 12):  # 保证有12个月特征 + 1个月标签
        idx = num_data
        num_data += 1

        # 日期判断：用于训练/测试划分
        predict_date = dates[i + 12]  # 第13个月的日期
        print(f"Processing sample {idx} for date {predict_date}...")
        data_type[idx] = 1 if predict_date >= np.datetime64("2017-07-01") else 0
        print(f"Data type for sample {idx}: {'Test' if data_type[idx] == 1 else 'Train'}")
        # 特征数据存入 data：12x5 矩阵
        data[idx] = features[i : i + 12]  # shape (12, 5)

        # 标签收益率 r（第13月开盘 - 第12月收盘）/ 第12月收盘
        close_prev = features[i + 11][3]  # 第12月收盘价
        open_next = features[i + 12][0]   # 第13月开盘价
        r = (open_next - close_prev) / close_prev

        windows_by_date[predict_date].append((r, idx))

print("✅ 数据处理完成，开始生成样本...")
# 横截面排序并分配标签
for date, lst in windows_by_date.items():
    print(f"Processing date {date} with {len(lst)} samples...")
    lst_sorted = sorted(lst, key=lambda x: x[0], reverse=True)
    for rank, (_, idx) in enumerate(lst_sorted):
        labels[idx] = rank/len(lst_sorted)  # 标签为排序后的百分比排名

# 转换为 numpy 数组（按 idx 排序）
sorted_idx = sorted(data.keys())
X = np.array([data[i] for i in sorted_idx])         # shape: (num_samples, 12, 5)
y = np.array([labels[i] for i in sorted_idx])       # shape: (num_samples,)
dtype = np.array([data_type[i] for i in sorted_idx])# shape: (num_samples,)
print(y.mean())
print(X.shape, y.shape, dtype.shape)
# 保存到 .npy 文件
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)
np.save(os.path.join(output_dir, "data_type.npy"), dtype)

print(f"✅ 总共生成样本数: {len(X)}，训练集: {np.sum(dtype == 0)}，测试集: {np.sum(dtype == 1)}")
