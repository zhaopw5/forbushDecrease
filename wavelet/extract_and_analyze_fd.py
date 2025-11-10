"""
Forbush Decrease AMS 小时分辨率数据提取与小波分析脚本

功能：
1. 从原始数据中提取指定时间范围的AMS数据
2. 创建FD_{日期}文件夹并保存数据
3. 进行小波分析并生成可视化结果

参考wavelet_v250923.py的调用方式，适配hourly分辨率数据
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta

# 导入小波分析函数
from ams_wavelet import waveletAnalysis

# ============== 配置参数 ==============
rig_bins = [1, 1.16, 1.33, 1.51, 1.71, 1.92, 2.15, 2.4, 2.67, 2.97,
            3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 6.47, 7.09, 7.76,
            8.48, 9.26, 10.1, 11, 13, 16.6, 22.8, 33.5, 48.5, 69.7, 100]

# 自定义颜色条的刻度格式
def scientific_notation_formatter(value, _):
    if value == 1:
        return "1"
    return r"$10^{{{}}}$".format(int(np.log10(value)))

# ============== 数据路径配置 ==============
RAW_DATA_FILE = "/home/zpw/Files/forbushDecrease/raw_data/ams/flux_long.csv"
FD_OUTPUT_PATH = "/home/zpw/Files/forbushDecrease/FDs/"

print("=" * 70)
print("Forbush Decrease (FD) 小波分析 - 小时分辨率")
print("=" * 70)

# ==================== 配置要分析的时间范围 ====================
start_date = datetime(2017, 3, 1)
end_date = datetime(2017, 12, 1)
# ================================================================

start_date_str = start_date.strftime('%Y_%m_%d')
end_date_str = end_date.strftime('%Y_%m_%d')
fd_folder_name = f"FD_{start_date.strftime('%Y%m%d')}"

# 1. 读取原始数据
print(f"\n读取原始AMS数据: {RAW_DATA_FILE}")
df_all = pd.read_csv(RAW_DATA_FILE)
df_all['date'] = pd.to_datetime(df_all['date'])
print(f"  总数据行数: {len(df_all)}")
print(f"  时间范围: {df_all['date'].min()} 到 {df_all['date'].max()}")

# 2. 提取指定时间范围
print(f"\n提取时间范围: {start_date.date()} 到 {end_date.date()}")
mask = (df_all['date'] >= start_date) & (df_all['date'] <= end_date)
df_extracted = df_all[mask].copy()
print(f"  提取数据行数: {len(df_extracted)}")

# 3. 创建输出文件夹
output_folder = os.path.join(FD_OUTPUT_PATH, fd_folder_name)
os.makedirs(output_folder, exist_ok=True)
print(f"\n创建输出文件夹: {output_folder}")

# 4. 保存提取的原始数据
raw_data_file = os.path.join(output_folder, "ams_data_extracted.csv")
df_extracted.to_csv(raw_data_file, index=False)
print(f"  原始数据已保存: {raw_data_file}")
