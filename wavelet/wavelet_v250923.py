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



rig_bins = [1,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,
			3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76,
			8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

# 自定义颜色条的刻度格式
def scientific_notation_formatter(value, _):
    if value == 1:
        return "1"  # 如果是 10^0，直接返回 1
    return r"$10^{{{}}}$".format(int(np.log10(value)))


#%% 小波分析结果用来画图
dataframe_dir = './results/flux_period_power'
os.makedirs(dataframe_dir, exist_ok=True)

# 画图保存路径
plot_dir = './results/plots_wavelet'
os.makedirs(plot_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 25


base_path = "/home/zpw/Files/AMS_NeutronMonitors/AMS/paperPlots/data"
version = 'version_3'


file_name = '2011-01-01-2024-07-31_pred_ams'
# file_name = '2014-07-01-2015-05-01_pred_ams'

# 从文件名解析起始日期
try:
    # 提取文件名中的日期部分（如 '2011-01-01-2024-07-31'）
    date_range_str = file_name.split("_")[0]
    # 分割并提取起始日期（如 '2011-01-01'）
    start_date = "-".join(date_range_str.split("-")[:3])
except IndexError:
    raise ValueError("文件名格式错误，应为 YYYY-MM-DD-YYYY-MM-DD_pred_ams")

# 读取AI预测的数据
file_path = os.path.join(base_path, "lightning_logs", version, f"{file_name}.csv")
original_df = pd.read_csv(file_path)

# 创建原始数据框的副本
df = original_df.copy()

# 生成日期列（自动匹配行数）
df.insert(0, "date", pd.date_range(start=start_date, freq="D", periods=len(df)))
df['date'] = pd.to_datetime(df['date'])
# 保存更新后的数据到新文件
df.to_csv(os.path.join(base_path, "lightning_logs", version, f'{file_name}_updated.csv'), index=False)


# AMS 已发布数据
file_path_AMS = os.path.join(base_path, 'table-s1-s2824.csv')
df_AMS = pd.read_csv(file_path_AMS)
df_AMS['date YYYY-MM-DD'] = pd.to_datetime(df_AMS['date YYYY-MM-DD'])


# 去除 SEP 的预测数据
rigidity_ranges = list(zip(rig_bins[:-1], rig_bins[1:]))

# 获取唯一日期:AMS有数据的日期（不包含探测器升级改造停机的日子）
unique_dates = df_AMS['date YYYY-MM-DD'].unique()
# 创建日期和刚度范围的笛卡尔积：有数据的日期*刚度范围
date_rigidity_combinations = pd.DataFrame(
    list(product(unique_dates, rigidity_ranges)),
    columns=['date YYYY-MM-DD', 'rigidity_range']
)
# 拆分刚度范围列为 min 和 max 列
date_rigidity_combinations[['rigidity_min GV', 'rigidity_max GV']] = pd.DataFrame(
    date_rigidity_combinations['rigidity_range'].tolist(),
    index=date_rigidity_combinations.index
)
# 去掉刚度范围列
date_rigidity_combinations.drop(columns=['rigidity_range'], inplace=True)
# 找出笛卡尔积和原始数据的不同:筛选出：那些高能有，低能没有的日期
original_keys = df_AMS[['date YYYY-MM-DD', 'rigidity_min GV', 'rigidity_max GV']]
diff = pd.merge(
    date_rigidity_combinations,
    original_keys,
    on=['date YYYY-MM-DD', 'rigidity_min GV', 'rigidity_max GV'],
    how='left',
    indicator=True
).query('_merge == "left_only"').drop(columns=['_merge'])
# diff保存了哪些日期，哪个低能 没有数据（这是SEP）

# 读AI预测的数据,把那些SEP日期的数据替换为 NaN
predicted_df = df.copy() # 这个copy非常重要，需要在副本上操作，否则下面的.loc会修改df

# 替换无意义的数据为 NaN
for _, row in diff.iterrows():
    # 找到刚度最小值在 rig_bins 中的序号
    rigidity_min = row['rigidity_min GV']
    column_index = rig_bins.index(rigidity_min)  # 获取序号
    column_name = str(column_index+1)  # 序号对应预测文件的列名
    
    # 根据日期找到对应行，替换值为 NaN
    predicted_df.loc[predicted_df['date'] == row['date YYYY-MM-DD'], column_name] = np.nan

df_noSEP = predicted_df.copy()


# 预测的相对误差
# df_error = pd.read_csv(os.path.join(base_path, f"rig_error_{version}.csv"))
df_error = pd.read_csv(os.path.join(base_path, "maximum_error_data.csv"))


# Bartels Rotation 日期
file_path_BR = os.path.join(base_path, 'bartels_rotation_number.csv')
df_BR = pd.read_csv(file_path_BR)
df_BR['start_date'] = pd.to_datetime(df_BR['start_date'])


# rig_bins = [1,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97, 3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76, 8.48,9.26,10.1,11,13,16.6,22.8]
rig_bins_short = [1, 2.15, 5.9, 9.26, 16.6]

# 选取时间段
date_list = [[datetime(2014, 1, 1), datetime(2016, 1, 1)]]
# date_list = [[datetime(2011, 1, 1), datetime(2024, 8, 1)]]
# date_list = [[datetime(2014, 3, 1), datetime(2015, 10, 29)]]
plot_xlim = [datetime(2014, 7, 1), datetime(2015, 5, 1)]
plot_begin_str = plot_xlim[0].strftime('%Y_%m_%d')
plot_end_str = plot_xlim[1].strftime('%Y_%m_%d')


# 定义颜色映射范围
vmin, vmax, number = 1e-1, 1e1, 20  # 颜色标尺范围

levels = np.logspace(np.log10(vmin), np.log10(vmax), number)  # 等高线分级
norm = BoundaryNorm(boundaries=np.logspace(np.log10(vmin), np.log10(vmax), number+1), ncolors=256, extend='both')


for wavelet_begin, wavelet_end in date_list:
    print('----------------')
    print(wavelet_begin, wavelet_end)

    # cut by date
    df_wavelet = df[(df['date'] >= wavelet_begin) & (df['date'] <= wavelet_end)].copy()# df.copy()#
    df_noSEP_plot = df_noSEP[(df_noSEP['date'] >= wavelet_begin) & (df_noSEP['date'] <= wavelet_end)].copy()
    df_AMS_plot = df_AMS[(df_AMS['date YYYY-MM-DD'] >= wavelet_begin) & (df_AMS['date YYYY-MM-DD'] <= wavelet_end)].copy()
    df_BR_plot = df_BR[(df_BR['start_date'] >= wavelet_begin) & (df_BR['start_date'] <= wavelet_end)].copy()

    wavelet_begin_str = wavelet_begin.strftime('%Y_%m_%d')
    wavelet_end_str = wavelet_end.strftime('%Y_%m_%d')
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
        rig_index = rig_bins.index(rig)
        rig_range_str = f"[{rig_bins[rig_index]:.2f}-{rig_bins[rig_index + 1]:.2f}]GV"

        # cut by rigidity
        rela_err = df_error[df_error['rig_min'] == rig]['std_dev'].values[0]
        flux_wavelet = df_wavelet[str(rig_index+1)].values
        flux_noSEP_plot = df_noSEP_plot[str(rig_index+1)].values
        df_ams = df_AMS_plot[df_AMS_plot['rigidity_min GV'] == rig]

        time = df_wavelet['date'].values

        _, period, power, coi, sig95, global_ws, global_signif = waveletAnalysis(flux_wavelet)

        # 对 power 值进行裁剪只映射一部分，范围之外统一颜色显示
        power_clipped = np.clip(power, vmin, vmax)

        power_df = pd.DataFrame(power, index=period, columns=time)
        # power_df_transposed = power_df.T
        filename = os.path.join(dataframe_dir, f"period_power_rig_{i}_{wavelet_begin_str}_{wavelet_end_str}_{version}.csv")
        power_df.to_csv(filename)

        # title_left = f'{wavelet_begin.strftime("%b %d, %Y")} - {wavelet_end.strftime("%b %d, %Y")}'
        title_left = f'{plot_xlim[0].strftime("%b %d, %Y")} - {plot_xlim[1].strftime("%b %d, %Y")}'
        title_right = 'Wavelet Power Spectrum'

        axs[i, 0].errorbar(x=df_noSEP_plot['date'].values, y=flux_noSEP_plot, yerr=flux_noSEP_plot * rela_err, c='red', fmt='.', label='Neutron Monitor')
        axs[i, 0].errorbar(x=df_ams['date YYYY-MM-DD'].values, y=df_ams['proton_flux m^-2sr^-1s^-1GV^-1'], 
                           yerr=df_ams['proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1'], 
                           c='blue', fmt='.',# markerfacecolor='none',
                           label=f'AMS')
        # 在每一个Bartels Rotation的开始日期画一条竖虚线
        for j in range(len(df_BR_plot)):
            axs[i, 0].axvline(df_BR_plot.iloc[j]['start_date'], color='black',linestyle=(0, (5, 10)), linewidth=1.)
        # axs[i, 0].tick_params(axis='x', rotation=15)
        axs[i, 0].set_xlim(plot_xlim[:])
        # axs[i, 0].set_xlim(wavelet_begin, wavelet_end)

        if i == 2:
            axs[i, 0].set_ylabel(r'Cosmic Ray Proton Flux $\mathrm{[m^{-2}s^{-1}sr^{-1}GV^{-1}]}$', labelpad=20)

        # axs[0, 0].legend(loc='upper right')

        if i == 0:
            axs[i, 0].set_title(title_left)

        # 调整 x 轴刻度标签与 x 轴之间的距离，单位为点
        axs[i, 0].tick_params(axis='x', pad=10)

        axs[i, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # 使用 MaxNLocator 设置最多显示5个刻度
        # axs[i, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        # 手动设置刻度位置，每隔两个日期设置一个刻度
        axs[i, 0].xaxis.set_ticks(df_noSEP_plot['date'].values[::70])

        # 获取当前的x轴刻度位置和标签
        ticks_loc = axs[i, 0].get_xticks()
        tick_labels = [mdates.num2date(t).strftime("%b %d \n %Y") for t in ticks_loc]

        # # 只设置第一个和最后一个刻度标签以包含年份
        # if len(ticks_loc) > 0:
        #     tick_labels[0] = mdates.num2date(ticks_loc[0]).strftime("%b %d \n %Y")
        #     tick_labels[-1] = mdates.num2date(ticks_loc[-1]).strftime("%b %d \n %Y")

        # 设置自定义的刻度标签
        axs[i, 0].set_xticks(ticks_loc)
        axs[i, 0].set_xticklabels(tick_labels)

        axs[i, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axs[0, 0].text(0.008, 0.85, 'Calculation', transform=axs[0, 0].transAxes, fontsize=30, color="red", ha="left")
        axs[0, 0].text(0.008, 0.68, 'AMS', transform=axs[0, 0].transAxes, fontsize=30, color="blue", ha="left")

        axs[i, 0].text(0.995, 0.85, rig_range_str, transform=axs[i, 0].transAxes, fontsize=35, fontweight='bold', color="black", ha="right")

        # CS = axs[i, 1].contourf(time, period, power, levels=20, cmap='jet')
        # 使用对数等高线图
        CS = axs[i, 1].contourf(time, period, power_clipped, levels=levels, cmap='jet', norm=norm)

        # axs[i, 1].contour(time, period, sig95, [-99, 1], colors='k')

        # axs[i, 1].fill_between(time, coi * 0 + period[-1], coi, facecolor="none", edgecolor="#00000040", hatch='x', alpha=0.5)

        # axs[i, 1].plot(time, coi, 'k', alpha=0.5)

        # 调整 x 轴刻度标签与 x 轴之间的距离，单位为点
        axs[i, 1].tick_params(axis='x', pad=10)

        # axs[i, 1].tick_params(axis='x', rotation=15)
        if i == 2:
            axs[i, 1].set_ylabel('Period (days)', labelpad=5)

        if i == 0:
            axs[i, 1].set_title(title_right)

        axs[i, 1].set_xlim(plot_xlim[:])
        # axs[i, 1].set_xlim(wavelet_begin, wavelet_end)

        axs[i, 1].set_yscale('log', base=2)

        # axs[i, 1].set_ylim([np.min(period), np.max(period)])
        axs[i, 1].set_ylim([np.max(period), np.min(period)])  # 倒序排列

        axs[i, 1].yaxis.set_major_formatter(ticker.ScalarFormatter())

        # axs[i, 1].ticklabel_format(axis='y', style='plain')

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

    fig.savefig(os.path.join(plot_dir, f'wavelet_{wavelet_begin_str}_{wavelet_end_str}_{plot_begin_str}_{plot_end_str}_{version}_new.pdf'))
    fig.savefig(os.path.join(plot_dir, f'wavelet_{wavelet_begin_str}_{wavelet_end_str}_{plot_begin_str}_{plot_end_str}_{version}_new.png'))

    plt.close(fig)