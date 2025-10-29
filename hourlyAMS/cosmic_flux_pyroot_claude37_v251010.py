import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ROOT

# 刚度区间定义
rig_bin_edges = [1,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,
                3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76,
                8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

def create_datetime_range():
    start = datetime(2011, 1, 1)
    end = datetime(2024, 7, 31, 0)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(hours=1)
    return dates

def find_rigidity_index(rigidity):
    """查找给定刚度值对应的索引"""
    for i, edge in enumerate(rig_bin_edges[:-1]):
        if abs(edge - rigidity) < 1e-6:  # 使用小数误差范围进行比较
            return i
    raise ValueError(f"刚度值 {rigidity} 未找到匹配的区间")

def load_and_process_data(flux_file, error_file):
    # 读取流强数据
    flux_data = pd.read_csv(flux_file, header=0)
    
    # 添加日期时间列
    dates = create_datetime_range()
    flux_data['datetime'] = dates
    
    # 读取误差数据
    error_data = pd.read_csv(error_file)
    
    return flux_data, error_data

def load_daily_data(daily_file):
    """加载每日数据"""
    daily_data = pd.read_csv(daily_file)
    daily_data['datetime'] = pd.to_datetime(daily_data['date YYYY-MM-DD'])
    return daily_data

def process_daily_data(daily_data, rigidity):
    """处理特定刚度区间的每日数据"""
    mask = (daily_data['rigidity_min GV'] == rigidity)
    data = daily_data[mask].copy()
    flux = data['proton_flux m^-2sr^-1s^-1GV^-1']
    error = data['proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1']
    dates = data['datetime']
    return dates, flux, error

def datetime_to_root_time(dt):
    """将 Python datetime 对象转换为 ROOT 时间格式"""
    # 将Python datetime转为Unix时间戳（秒）
    timestamp = int(dt.timestamp())
    # 直接返回Unix时间戳
    return timestamp

def plot_relative_flux_with_error_root(flux_data, error_data, rigidity, daily_data=None, time_range=None, save_dir='flux_comparison_plots', normalize=False, norm_baseline='mean'):
    """
    使用 PyROOT 绘制指定刚度区间的相对流强变化随时间变化图
    
    Parameters:
    -----------
    normalize : bool, optional
        是否进行归一化绘图 (default: False)
    norm_baseline : str, optional
        归一化基线类型，可选 'mean', 'median', 'first' (default: 'mean')
    """
    # 设置 ROOT 绘图风格
    ROOT.gStyle.SetOptStat(0)
    
    # 创建画布
    c = ROOT.TCanvas(f"c_rel_{int(rigidity*100)}", f"", 1200, 800)
    
    # 设置画布背景为白色
    c.SetFillColor(ROOT.kWhite)
    
    # 设置时间格式（全局设置）
    ROOT.gStyle.SetTimeOffset(0)
    
    # 获取刚度区间索引
    rig_idx = find_rigidity_index(rigidity)
    
    # 处理每小时数据
    flux = flux_data.iloc[:, rig_idx]
    dates = flux_data['datetime']
    rel_error = error_data.iloc[rig_idx]['std_dev']
    
    # 应用时间范围过滤
    if time_range:
        mask = (dates >= time_range[0]) & (dates <= time_range[1])
        dates = dates[mask]
        flux = flux[mask]
    
    # 计算绝对误差
    error = np.abs(rel_error * flux)
    
    # 归一化处理
    original_flux = flux.copy()  # 保留原始流强数据
    if normalize:
        # 计算归一化基线
        if norm_baseline == 'mean':
            baseline = flux.mean()
        elif norm_baseline == 'median':
            baseline = flux.median()
        elif norm_baseline == 'first':
            baseline = flux.iloc[0]
        else:
            raise ValueError(f"不支持的归一化基线类型: {norm_baseline}")
        
        # 归一化流强和误差
        flux = flux / baseline
        error = error / baseline
    
    # 转换日期为 ROOT 时间戳
    date_timestamps = [datetime_to_root_time(d) for d in dates]
    
    # 创建 TGraphErrors 对象
    n_points = len(date_timestamps)
    gr_hourly = ROOT.TGraphErrors(n_points)
    
    for i in range(n_points):
        gr_hourly.SetPoint(i, date_timestamps[i], flux.iloc[i])
        gr_hourly.SetPointError(i, 0, error.iloc[i])
    
    # 设置图表属性
    hourly_label = f'Hourly'  # 移除刚度区间信息
    gr_hourly.SetTitle("")
    gr_hourly.SetMarkerStyle(20)
    gr_hourly.SetMarkerSize(0.4)  # 稍小的标记
    gr_hourly.SetMarkerColor(ROOT.kBlue)
    gr_hourly.SetLineColor(ROOT.kBlue)
    
    # 设置时间显示格式
    gr_hourly.GetXaxis().SetTimeDisplay(1)
    # 修改时间格式：只显示日
    gr_hourly.GetXaxis().SetTimeFormat("%d")
    gr_hourly.GetXaxis().SetTimeOffset(0, "gmt")  # 关键：使用GMT时区并且无偏移
    gr_hourly.GetXaxis().SetLabelSize(0.04)  # 增大x轴标签字号
    gr_hourly.GetYaxis().SetLabelSize(0.04)  # 增大y轴标签字号
    gr_hourly.GetXaxis().SetTitleSize(0.04)  # 增大x轴标题字号
    gr_hourly.GetYaxis().SetTitleSize(0.04)  # 增大y轴标题字号
    gr_hourly.GetXaxis().SetTitleOffset(1.4)  # 增加X轴标题与X轴的距离
    gr_hourly.GetYaxis().SetTitleOffset(1.2)  # 增加Y轴标题与Y轴的距离
    gr_hourly.GetYaxis().SetLabelOffset(0.01)  # Y轴标签偏移
    gr_hourly.GetXaxis().SetLabelOffset(0.015)  # 增加X轴标签偏移，增加与下边框距离
    gr_hourly.GetXaxis().SetTitle("")
    if normalize:
        gr_hourly.GetYaxis().SetTitle("Normalized Flux")
    else:
        gr_hourly.GetYaxis().SetTitle("Proton Flux (m^{-2}sr^{-1}s^{-1}GV^{-1})")
    
    # 设置y轴标题居中对齐
    gr_hourly.GetYaxis().CenterTitle(ROOT.kTRUE)
    
    # 设置坐标轴范围与数据范围相同
    if time_range:
        start_time = datetime_to_root_time(time_range[0])
        end_time = datetime_to_root_time(time_range[1])
        gr_hourly.GetXaxis().SetRangeUser(start_time, end_time)
    
    # 设置每天一个刻度标签 - 对于31天的时间范围，关闭小刻度
    gr_hourly.GetXaxis().SetNdivisions(820)
    
    # 设置坐标轴颜色为黑色
    gr_hourly.GetXaxis().SetAxisColor(ROOT.kBlack)
    gr_hourly.GetYaxis().SetAxisColor(ROOT.kBlack)
    gr_hourly.GetXaxis().SetLabelColor(ROOT.kBlack)
    gr_hourly.GetYaxis().SetLabelColor(ROOT.kBlack)
    gr_hourly.GetXaxis().SetTitleColor(ROOT.kBlack)
    gr_hourly.GetYaxis().SetTitleColor(ROOT.kBlack)
    gr_hourly.GetXaxis().SetTickLength(0.02)
    gr_hourly.GetYaxis().SetTickLength(0.02)
    
    # 设置画布边距
    c.SetTopMargin(0.01)
    c.SetBottomMargin(0.12)  # 增加底部边距以容纳调整后的x轴标签
    c.SetRightMargin(0.05)
    c.SetLeftMargin(0.1)  # 增加左边距以容纳Y轴标签

    # 绘制图表
    gr_hourly.Draw("AP")
    
    # 设置坐标框背景和边框颜色
    c.SetFrameFillColor(ROOT.kWhite)  # 坐标框背景白色
    c.SetFrameLineColor(ROOT.kBlack)  # 坐标框边框黑色
    
    # 精确设置x轴范围，确保严格对应时间范围，不留空白
    if time_range:
        start_time = datetime_to_root_time(time_range[0])
        end_time = datetime_to_root_time(time_range[1])
        # 使用SetLimits设置硬限制，确保不留空白
        gr_hourly.GetXaxis().SetLimits(start_time, end_time)
        # 再用SetRangeUser确保显示范围
        gr_hourly.GetXaxis().SetRangeUser(start_time, end_time)
        # 强制更新画布
        c.Modified()
        c.Update()

    # 处理每日数据
    if daily_data is not None:
        daily_dates, daily_flux, daily_error = process_daily_data(daily_data, rig_bin_edges[rig_idx])
        
        if time_range:
            mask = (daily_dates >= time_range[0]) & (daily_dates <= time_range[1])
            daily_dates = daily_dates[mask]
            daily_flux = daily_flux[mask]
            daily_error = daily_error[mask]
        
        # 归一化每日数据（使用与每小时数据相同的基线）
        if normalize:
            # 使用与每小时数据相同的基线进行归一化
            daily_flux = daily_flux / baseline
            daily_error = daily_error / baseline
        
        # 转换日期为 ROOT 时间戳
        daily_timestamps = [datetime_to_root_time(d) for d in daily_dates]
        
        # 创建每日数据的 TGraphErrors
        n_daily = len(daily_timestamps)
        gr_daily = ROOT.TGraphErrors(n_daily)
        
        for i in range(n_daily):
            gr_daily.SetPoint(i, daily_timestamps[i], daily_flux.iloc[i])
            gr_daily.SetPointError(i, 0, daily_error.iloc[i])
        
        # 设置每日图表属性
        daily_label = f'Daily'  # 移除刚度区间信息
        gr_daily.SetMarkerStyle(21)
        gr_daily.SetMarkerSize(0.8)
        gr_daily.SetMarkerColor(ROOT.kRed+1)  # 使用稍深的红色
        gr_daily.SetLineColor(ROOT.kRed+1)
        
        # 绘制每日图表
        gr_daily.Draw("P SAME")

    
    # 添加刚度区间标签（黑色，只显示一次）
    rigidity_label = f'[{rig_bin_edges[rig_idx]:.2f}-{rig_bin_edges[rig_idx+1]:.2f}] GV'
    text_rigidity = ROOT.TLatex()
    text_rigidity.SetNDC()
    text_rigidity.SetTextFont(42)
    text_rigidity.SetTextSize(0.05)  # 稍大的字体作为主标签
    text_rigidity.SetTextColor(ROOT.kBlack)  # 黑色
    text_rigidity.DrawLatex(0.15, 0.15, rigidity_label)  # 统一位置
    
    # 添加数据类型标签
    text_hourly = ROOT.TLatex()
    text_hourly.SetNDC()
    text_hourly.SetTextFont(42)
    text_hourly.SetTextSize(0.05)  # 稍小的字体
    text_hourly.SetTextColor(ROOT.kBlue)
    text_hourly.DrawLatex(0.85, 0.93, "Hourly")
    
    if daily_data is not None and len(daily_dates) > 0:
        text_daily = ROOT.TLatex()
        text_daily.SetNDC()
        text_daily.SetTextFont(42)
        text_daily.SetTextSize(0.05)  # 稍小的字体
        text_daily.SetTextColor(ROOT.kRed+1)
        text_daily.DrawLatex(0.85, 0.88, "Daily")
    
    c.cd()
    if time_range:
        year_text = ROOT.TLatex()
        year_text.SetNDC()
        year_text.SetTextFont(42)
        year_text.SetTextSize(0.05)
        year_text.SetTextColor(ROOT.kBlack)  # 年份标签设为黑色
        # 合并显示月份和年份
        year_text.DrawLatex(0.5, 0.02, "March 2015")
        
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用刚度值作为文件名
    filename = f'flux_comparison_{rig_bin_edges[rig_idx]}-{rig_bin_edges[rig_idx+1]}GV'
    save_path = os.path.join(save_dir, filename)
    
    # 保存图像
    c.SaveAs(f"{save_path}.png")
    c.SaveAs(f"{save_path}.pdf")
    
    return save_path

if __name__ == "__main__":
    # 文件路径
    flux_file = "STL_2011-01-01-2024-07-31_hourly_FDs_pred_ams.csv"
    # error_file = "rig_error_version_3.csv"
    error_file = "maximum_error_data.csv"
    daily_file = "table-s1-s2824.csv"
    
    # 加载数据
    flux_data, error_data = load_and_process_data(flux_file, error_file)
    daily_data = load_daily_data(daily_file)
    
    # 设置时间范围（可选）
    time_range = (
        datetime(2015, 3, 10),
        datetime(2015, 3, 24)
    )

    
    # 创建保存目录
    save_dir = 'flux_comparison_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置ROOT全局时间偏移（重要！）
    ROOT.gStyle.SetTimeOffset(0)
    
    # 处理所有刚度区间的相对流量变化
    saved_files = []
    for rigidity in rig_bin_edges[:-1]:  # 遍历除最后一个值外的所有刚度值
        save_path = plot_relative_flux_with_error_root(
            flux_data, 
            error_data, 
            rigidity, 
            daily_data, 
            time_range,
            save_dir,
            normalize=False,  # 设置为True启用归一化
            norm_baseline='mean'  # 可选：'mean', 'median', 'first'
        )
        saved_files.append(save_path)
    
    print(f"已生成 {len(saved_files)} 张图片，保存在 {save_dir} 目录下")
