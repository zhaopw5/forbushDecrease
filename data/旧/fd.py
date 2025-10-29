# -*- coding: utf-8 -*-
"""
For each AMS rigidity bin, plot ΔI(t) together with key OMNI parameters:
|B|, Bz_gsm, Vsw, Np, Dst, AE.

Each rigidity gets its own multi-panel figure (6 subplots, share x-axis).
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- path ----------
CSV_PATH = "merged_FD_20150619_20150707.csv"
OUT_DIR  = "plots_by_rigidity"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- read ----------
df = pd.read_csv(CSV_PATH, parse_dates=["datetime"]).set_index("datetime")

# identify rigidity columns
rig_cols = [c for c in df.columns if "GV" in c and not c.endswith("_err")]

# select omni columns to plot (你可以按需要改顺序或删减)
omni_cols = ["B_avg_abs", "Bz_gsm", "Vsw", "Np", "Dst", "AE"]
labels = ["|B| [nT]", "Bz GSM [nT]", "Vsw [km/s]", "Np [cm⁻³]",
          "Dst [nT]", "AE [nT]"]
colors = ["tab:red", "tab:orange", "tab:blue", "tab:green",
          "tab:purple", "tab:brown"]

# ---------- iterate each rigidity ----------
for rig in rig_cols:
    fig, axes = plt.subplots(len(omni_cols)+1, 1,
                             figsize=(10, 9),
                             sharex=True,
                             gridspec_kw={"hspace": 0.1})

    # 1) ΔI
    ax0 = axes[0]
    ax0.plot(df.index, df[rig], color='black', lw=1.2)
    ax0.axhline(0, color='gray', lw=0.8, ls='--')
    ax0.set_ylabel(f"ΔI\n({rig})")
    ax0.grid(True, ls='--', alpha=0.4)

    # 2) OMNI parameters
    for i, (col, lab, c) in enumerate(zip(omni_cols, labels, colors)):
        ax = axes[i+1]
        ax.plot(df.index, df[col], color=c, lw=1.0)
        ax.set_ylabel(lab, rotation=0, labelpad=40)
        ax.grid(True, ls='--', alpha=0.4)

    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"AMS ΔI & Solar Parameters — {rig}", fontsize=13)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT_DIR}/fd_panel_{rig.replace(' ', '').replace('/', '-')}.png", dpi=300)
    plt.close(fig)

print(f"[OK] generated {len(rig_cols)} multi-panel plots in {OUT_DIR}/")
