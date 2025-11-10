import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 全局字体放大与加粗
plt.rcParams.update({
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ----------------------------
# 配置
# ----------------------------
AMS_FILE = "ams_processed.csv"
OMNI_FILE = "omni_processed.csv"
OUTPUT_DIR = "plots_ams_I_vs_omni"
OMNI_VARS = ["B_avg_abs", "Bz_gse", "Np", "Vsw"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 读取数据
# ----------------------------
ams = pd.read_csv(AMS_FILE, parse_dates=["datetime"])
ams.set_index("datetime", inplace=True)

omni = pd.read_csv(OMNI_FILE, parse_dates=["datetime"])
omni.set_index("datetime", inplace=True)

# 时间对齐：取交集
common_index = ams.index.intersection(omni.index)
ams = ams.loc[common_index]
omni = omni.loc[common_index]

# ----------------------------
# 提取刚度通道：基于 I_X-YGV 列
# ----------------------------
I_cols = [col for col in ams.columns if col.startswith("I_") and col.endswith("GV") and "_err" not in col]
rigidities = [col.replace("I_", "").replace("GV", "") for col in I_cols]

print(f"Found {len(rigidities)} rigidity channels: {rigidities}")

# ----------------------------
# 绘图
# ----------------------------
for I_col, rig in zip(I_cols, rigidities):
    err_col = I_col + "_err"
    if err_col not in ams.columns:
        print(f"[WARN] Error column {err_col} not found. Skipping {rig} GV.")
        continue

    # 提取数据
    x = ams.index
    y = ams[I_col]
    yerr = ams[err_col]

    # 创建四行一列子图
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    # fig.suptitle(f"AMS Absolute Intensity (I) vs OMNI Parameters\nRigidity: {rig} GV", fontsize=14)
    # 设置各个子图之间的空隙=0
    plt.subplots_adjust(hspace=0)

    for i, var in enumerate(OMNI_VARS):
        if var not in omni.columns:
            # 缺失变量提示文本加粗并放大
            axes[i].text(0.5, 0.5, f"{var} not available", transform=axes[i].transAxes,
                         ha='center', fontweight='bold', fontsize=13)
            axes[i].set_ylabel(var, fontweight='bold')  # 标签加粗
            # y 轴刻度加粗
            for t in axes[i].get_yticklabels():
                t.set_fontweight('bold')
            continue

        # 绘制 OMNI 参数（左侧 y 轴）
        axes[i].plot(omni.index, omni[var], color='tab:blue', linewidth=1, label=var)
        axes[i].set_ylabel(var, color='tab:blue', fontsize=14, fontweight='bold')
        axes[i].tick_params(axis='y', labelcolor='tab:blue', labelsize=12, width=1.2)
        # y 轴刻度加粗
        for t in axes[i].get_yticklabels():
            t.set_fontweight('bold')
        axes[i].grid(True, linestyle='--', alpha=0.5)

        # 右侧 y 轴：AMS 绝对强度 + 误差棒
        ax2 = axes[i].twinx()
        ax2.errorbar(x, y, yerr=yerr, fmt='o', color='tab:red', ecolor='lightcoral',
                     elinewidth=0.8, capsize=1.5, markersize=2, alpha=0.8, label='AMS I')
        ax2.set_ylabel("AMS Flux", color='tab:red', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', colors='tab:red', labelsize=12, width=1.2)
        # 右侧 y 轴刻度加粗
        for t in ax2.get_yticklabels():
            t.set_fontweight('bold')

    # 共享 x 轴刻度放大并加粗（仅底部显示，但对所有轴设置更稳妥）
    for ax in axes:
        ax.tick_params(axis='x', labelsize=12, width=1.2, labelrotation=15)

        for t in ax.get_xticklabels():
            t.set_fontweight('bold')

    # axes[-1].set_xlabel("Datetime", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 安全文件名（替换 . 为 _）
    safe_rig = rig.replace(".", "_")
    output_path = os.path.join(OUTPUT_DIR, f"FD_20150619_ams_I_{safe_rig}GV_vs_omni.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

print(f"\n✅ All plots saved to '{OUTPUT_DIR}' directory.")