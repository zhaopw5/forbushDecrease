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

# 5. 创建小波分析结果文件夹
results_dir = os.path.join(output_folder, "wavelet_results")
os.makedirs(results_dir, exist_ok=True)

# 6. 创建颜色映射范围
vmin, vmax, number = 1e-1, 1e1, 20
levels = np.logspace(np.log10(vmin), np.log10(vmax), number)
norm = BoundaryNorm(boundaries=np.logspace(np.log10(vmin), np.log10(vmax), number+1), 
                    ncolors=256, extend='both')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 7. 重新组织数据：按刚度范围和时间重新索引
print(f"\n重新组织数据为按刚度范围的时间序列...")
rigidity_ranges = df_extracted.groupby(['rigidity_min', 'rigidity_max']).size().index.tolist()
print(f"  找到 {len(rigidity_ranges)} 个刚度范围")

# 创建一个dataframe，其中列是刚度范围，行是时间点，值是flux
# 先获取所有唯一时间点
unique_times_sorted = sorted(df_extracted['date'].unique())
print(f"  时间点数: {len(unique_times_sorted)}")

# 为每个刚度范围创建时间序列
data_by_rigidity = {}
for rig_min, rig_max in rigidity_ranges:
    mask_rig = (df_extracted['rigidity_min'] == rig_min) & (df_extracted['rigidity_max'] == rig_max)
    df_rig = df_extracted[mask_rig].copy()
    
    # 按时间排序
    df_rig = df_rig.sort_values('date')
    
    # 创建flux时间序列（与unique_times_sorted对应）
    flux_series = []
    for t in unique_times_sorted:
        mask_t = df_rig['date'] == t
        if mask_t.any():
            flux_series.append(df_rig[mask_t]['flux'].values[0])
        else:
            flux_series.append(np.nan)
    
    data_by_rigidity[(rig_min, rig_max)] = {
        'times': unique_times_sorted,
        'flux': np.array(flux_series)
    }

# 8. 对每个刚度范围进行小波分析
print(f"\n开始小波分析...")
print("-" * 70)

success_count = 0
for rig_index, (rig_min, rig_max) in enumerate(rigidity_ranges):
    rig_range_str = f"[{rig_min:.2f}-{rig_max:.2f}]GV"
    print(f"\n{rig_index+1:2d}. 刚度范围: {rig_range_str}")
    
    times = data_by_rigidity[(rig_min, rig_max)]['times']
    flux_series = data_by_rigidity[(rig_min, rig_max)]['flux']
    
    # 检查有效数据
    valid_count = np.sum(~np.isnan(flux_series))
    print(f"    数据点数: {len(flux_series)}, 有效数据: {valid_count}")
    
    if valid_count < 10:
        print(f"    跳过: 有效数据过少")
        continue
    
    # 调用小波分析函数（与参考代码相同的调用方式）
    try:
        _, period, power, coi, sig95, global_ws, global_signif = waveletAnalysis(flux_series)
        
        success_count += 1
        print(f"    ✓ 小波分析成功")
        print(f"      周期范围: {period.min():.2f} - {period.max():.2f} 天")
        
        # 保存功率数据
        power_df = pd.DataFrame(power, index=period, columns=times)
        rig_str = f"{rig_min:.2f}_{rig_max:.2f}".replace('.', 'p')
        power_file = os.path.join(results_dir, f"power_{rig_str}.csv")
        power_df.to_csv(power_file)
        print(f"      功率数据已保存")
        
        # 绘制图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                        gridspec_kw={'height_ratios': [1, 2.5]})
        
        # 上图：时间序列
        ax1.plot(times, flux_series, 'b-', linewidth=1, label='Flux')
        ax1.fill_between(times, flux_series, alpha=0.2)
        ax1.set_ylabel('Flux', fontsize=10)
        ax1.set_title(f'Cosmic Ray Flux: {rig_range_str}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 下图：小波功率谱
        time_numeric = mdates.date2num(times)
        power_clipped = np.clip(power, vmin, vmax)
        CS = ax2.contourf(time_numeric, period, power_clipped, levels=levels,
                         cmap='jet', norm=norm)
        
        ax2.set_ylabel('Period (days)', fontsize=10)
        ax2.set_xlabel('Time', fontsize=10)
        ax2.set_yscale('log', base=2)
        ax2.set_ylim([np.max(period), np.min(period)])
        ax2.set_title('Wavelet Power Spectrum', fontsize=12, fontweight='bold')
        
        # 颜色条
        cbar = plt.colorbar(CS, ax=ax2, aspect=15, fraction=0.12, pad=0.01)
        cbar.set_label('Normalized Power', fontsize=10)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
        
        # 格式化x轴时间标签
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图表
        rig_str = rig_range_str.replace('[', '').replace(']', '').replace('-', '_')
        fig_file = os.path.join(results_dir, f"wavelet_{rig_str}.png")
        plt.savefig(fig_file, dpi=100, bbox_inches='tight')
        print(f"      图表已保存")
        
        plt.close(fig)
        
    except Exception as e:
        print(f"    ✗ 小波分析失败: {e}")
        continue

print("\n" + "=" * 70)
print(f"分析完成！")
print(f"成功分析: {success_count}/{len(rigidity_ranges)} 个刚度范围")
print(f"结果保存位置: {output_folder}")
print("=" * 70)