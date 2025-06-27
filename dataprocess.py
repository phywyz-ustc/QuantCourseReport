import os
import pandas as pd

# 设置原始和目标路径
input_dir = r'D:\AI4Charting\大作业\individual_stocks_5yr'
output_dir = r'D:\AI4Charting\大作业\after'

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 CSV 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(input_dir, filename)
        print(f'正在处理: {filename}')
        # 读取 CSV
        df = pd.read_csv(filepath, parse_dates=['date'])

        # 加入 month 标签
        df['month'] = df['date'].dt.to_period('M')

        # 按 Name 和 month 获取每月最早的那天
        df_sorted = df.sort_values('date')
        first_day = df_sorted.groupby(['Name', 'month']).first().reset_index()

        # 保留最早日期 ≤ 7号 的数据
        first_day = first_day[first_day['date'].dt.day <= 7]

        # 去掉临时的 month 列
        first_day.drop(columns=['month'], inplace=True)

        # 保存处理后的 CSV
        output_path = os.path.join(output_dir, filename)
        first_day.to_csv(output_path, index=False)

        print(f'处理完成: {filename}')
