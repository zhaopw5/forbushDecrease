import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from itertools import product

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


# AMS 已发布daily数据
base_path = "/home/zpw/Files/AMS_NeutronMonitors/AMS/paperPlots/data"
df_daily_AMS = os.path.join(base_path, 'table-s1-s2824.csv')
df_AMS = pd.read_csv(df_daily_AMS)
df_AMS['date YYYY-MM-DD'] = pd.to_datetime(df_AMS['date YYYY-MM-DD'])


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
# cut daily ams data with start_date and end_date
df_AMS_extracted = df_AMS[(df_AMS['date YYYY-MM-DD'] >= start_date) & (df_AMS['date YYYY-MM-DD'] <= end_date)].copy()
print(f"  提取AMS daily数据行数: {len(df_AMS_extracted)}")

# 3. 创建输出文件夹
output_folder = os.path.join(FD_OUTPUT_PATH, fd_folder_name)
os.makedirs(output_folder, exist_ok=True)
print(f"\n创建输出文件夹: {output_folder}")

# 4. 保存提取的原始数据
raw_data_file = os.path.join(output_folder, "ams_data_extracted.csv")
df_extracted.to_csv(raw_data_file, index=False)
print(f"  原始数据已保存: {raw_data_file}")



#%% 小波分析结果用来画图
dataframe_dir = os.path.join(output_folder, "flux_period_power")
os.makedirs(dataframe_dir, exist_ok=True)

# 画图保存路径
plot_dir = os.path.join(output_folder, "plots_wavelet")
os.makedirs(plot_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 25


df_extracted['date'] = pd.to_datetime(df_extracted['date'])


rigidity_ranges = list(zip(rig_bins[:-1], rig_bins[1:]))

# 获取唯一日期:AMS有数据的日期（不包含探测器升级改造停机的日子）
unique_dates = df_extracted['date'].unique()

# 创建日期和刚度范围的笛卡尔积：有数据的日期*刚度范围
date_rigidity_combinations = pd.DataFrame(
    list(product(unique_dates, rigidity_ranges)),
    columns=['date', 'rigidity_range']
)
# 拆分刚度范围列为 min 和 max 列
date_rigidity_combinations[['rigidity_min', 'rigidity_max']] = pd.DataFrame(
    date_rigidity_combinations['rigidity_range'].tolist(),
    index=date_rigidity_combinations.index
)
# 去掉刚度范围列
date_rigidity_combinations.drop(columns=['rigidity_range'], inplace=True)


rig_bins_short = [1, 2.15, 5.9, 9.26, 16.6]


# 定义颜色映射范围
vmin, vmax, number = 1e-1, 1e1, 20  # 颜色标尺范围

levels = np.logspace(np.log10(vmin), np.log10(vmax), number)  # 等高线分级
norm = BoundaryNorm(boundaries=np.logspace(np.log10(vmin), np.log10(vmax), number+1), ncolors=256, extend='both')


wavelet_begin_str = start_date_str
wavelet_end_str = end_date_str
date_str = f"{wavelet_begin_str}_{wavelet_end_str}"

fig, axs = plt.subplots(5, 2, figsize=(25, 18), sharex=True,
                        gridspec_kw={'width_ratios': [1, 1.185]})
fig.subplots_adjust(hspace=0.15, wspace=0.12, left=0.07, right=0.99, top=0.95, bottom=0.06)

for ax in axs.flat:
    # 设置主刻度线朝内
    ax.tick_params(axis="both", direction="in", which="major", width=2, length=6,
                    top=True, bottom=True, left=True, right=True)
    # 设置次刻度线朝内
    ax.tick_params(axis="both", direction="in", which="minor", width=2, length=3,
                    top=True, bottom=True, left=True, right=True)
    # 设置四周边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2)


for i in range(len(rig_bins_short)):
    print('i=', i)
    rig = rig_bins_short[i]
    print('rig=', rig)
    
    rig_index = rig_bins.index(rig)
    rig_range_str = f"[{rig_bins[rig_index]:.2f}-{rig_bins[rig_index + 1]:.2f}]GV"
    print('rig_range_str=', rig_range_str)

    print("df_extracted_all:", df_extracted[:10])
    
    # cut by rigidity
    df_extracted_rig = df_extracted[df_extracted['rigidity_min'] == rig]
    print("df_extracted:", df_extracted[:5])
    # cut daily ams data with rigidity
    df_AMS_extracted_rig = df_AMS_extracted[df_AMS_extracted['rigidity_min GV'] == rig]
    print("df_AMS_extracted_rig:", df_AMS_extracted_rig[:5])

    time = df_extracted_rig['date'].values
    flux_wavelet = df_extracted_rig['flux'].values
    print("df_extracted_rig:", df_extracted_rig[:5])
    
    _, period, power, coi, sig95, global_ws, global_signif = waveletAnalysis(flux_wavelet)

    # 对 power 值进行裁剪只映射一部分，范围之外统一颜色显示
    power_clipped = np.clip(power, vmin, vmax)

    power_df = pd.DataFrame(power, index=period, columns=time)
    # power_df_transposed = power_df.T
    filename = os.path.join(dataframe_dir, f"period_power_rig_{i}.csv")
    power_df.to_csv(filename)

    title_right = 'Wavelet Power Spectrum'

    axs[i, 0].errorbar(x=df_extracted_rig['date'].values, y=df_extracted_rig['flux'], 
                        yerr=df_extracted_rig['error_bar'], 
                        c='blue', fmt='.',# markerfacecolor='none',
                        label=f'AMS')
    axs[i, 0].errorbar(x=df_AMS_extracted_rig['date YYYY-MM-DD'].values, 
                        y=df_AMS_extracted_rig['proton_flux m^-2sr^-1s^-1GV^-1'], 
                        yerr=df_AMS_extracted_rig['proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1'],
                        c='red', fmt='s', markersize=6, markerfacecolor='none',
                        label='AMS daily')

    if i == 2:
        axs[i, 0].set_ylabel(r'Cosmic Ray Proton Flux $\mathrm{[m^{-2}s^{-1}sr^{-1}GV^{-1}]}$', labelpad=20)


    # 调整 x 轴刻度标签与 x 轴之间的距离，单位为点
    axs[i, 0].tick_params(axis='x', pad=10)

    # axs[i, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # 使用 MaxNLocator 设置最多显示5个刻度
    # axs[i, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    # 手动设置刻度位置，每隔两个日期设置一个刻度
    axs[i, 0].xaxis.set_ticks(df_extracted_rig['date'].values[::500])

    # 获取当前的x轴刻度位置和标签
    ticks_loc = axs[i, 0].get_xticks()
    tick_labels = [mdates.num2date(t).strftime("%b \n %d") for t in ticks_loc]

    # 设置自定义的刻度标签
    axs[i, 0].set_xticks(ticks_loc)
    axs[i, 0].set_xticklabels(tick_labels)

    axs[i, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())

    axs[0, 0].text(0.008, 0.68, 'AMS', transform=axs[0, 0].transAxes, fontsize=30, color="blue", ha="left")

    axs[i, 0].text(0.995, 0.85, rig_range_str, transform=axs[i, 0].transAxes, fontsize=35, fontweight='bold', color="black", ha="right")

    # CS = axs[i, 1].contourf(time, period, power, levels=20, cmap='jet')
    # 使用对数等高线图
    CS = axs[i, 1].contourf(time, period, power_clipped, levels=levels, cmap='jet', norm=norm)

    axs[i, 1].contour(time, period, sig95, [-99, 1], colors='k')

    axs[i, 1].fill_between(time, coi * 0 + period[-1], coi, facecolor="none", edgecolor="#00000040", hatch='x', alpha=0.5)

    axs[i, 1].plot(time, coi, 'k', alpha=0.5)

    # 调整 x 轴刻度标签与 x 轴之间的距离，单位为点
    axs[i, 1].tick_params(axis='x', pad=10)

    # axs[i, 1].tick_params(axis='x', rotation=15)
    if i == 2:
        axs[i, 1].set_ylabel('Period (hours)', labelpad=5)

    if i == 0:
        axs[i, 0].set_title(f'{start_date_str}-{end_date_str} flux ')
        axs[i, 1].set_title(title_right)


    axs[i, 1].set_yscale('log', base=2)

    axs[i, 1].set_ylim([np.max(period), np.min(period)])  # 倒序排列

    axs[i, 1].yaxis.set_major_formatter(ticker.ScalarFormatter())

    axs[i, 1].text(0.995, 0.85, rig_range_str, transform=axs[i, 1].transAxes, fontsize=35, fontweight='bold', color="white", ha="right")#, fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(CS, 
                        ax=axs[i, 1], # 指定将颜色条附加到的子图）
                        aspect=10, # 控制颜色条的宽高比，值越大颜色条越瘦
                        fraction=0.15, # 颜色条占子图宽度的比例，值越小颜色条越窄
                        pad=0.01, # 颜色条与子图之间的距离，单位为图宽的比例
                        label='Normalized Power'
                        )
    # 调整标签与颜色条的距离
    cbar.ax.yaxis.labelpad = 15  # 设置纵向颜色条标签的距离

    cbar.locator = ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=10)
    # cbar.formatter = LogFormatter(base=10.0, labelOnlyBase=False)  # 使用内置的对数格式，not working
    cbar.formatter = FuncFormatter(scientific_notation_formatter)  # 使用自定义科学计数法格式
    cbar.update_ticks()

    # # 使用 MaxNLocator 减少颜色条上的标度数量
    # cbar.locator = ticker.MaxNLocator(nbins=5)
    # cbar.update_ticks()

fig.savefig(os.path.join(plot_dir, f'wavelet_{wavelet_begin_str}_{wavelet_end_str}.pdf'))
fig.savefig(os.path.join(plot_dir, f'wavelet_{wavelet_begin_str}_{wavelet_end_str}.png'))

plt.close(fig)