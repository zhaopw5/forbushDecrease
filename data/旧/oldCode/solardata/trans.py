import pandas as pd
import numpy as np

INPUT_ASC = "omni_min2015.asc"
OUTPUT_CSV = "omni_min2015.csv"

cols = [
    "Year", "DOY", "Hour", "Minute",
    "IMF_SC_ID", "SW_SC_ID",
    "Npts_IMF", "Npts_SW",
    "Percent_interp",
    "Timeshift_sec", "RMS_Timeshift_sec",
    "RMS_PFN",
    "DBOT1_sec",
    "B", "Bx_gse", "By_gse", "Bz_gse", "By_gsm", "Bz_gsm",
    "RMS_SD_B", "RMS_SD_Bvec",
    "V", "Vx_gse", "Vy_gse", "Vz_gse",
    "Np", "Tp", "Pdyn", "E_y_gsm", "Plasma_beta", "M_A",
    "X_gse_Re", "Y_gse_Re", "Z_gse_Re",
    "BSN_Xgse_Re", "BSN_Ygse_Re", "BSN_Zgse_Re",
    "AE", "AL", "AU", "SYM_D", "SYM_H", "ASY_D", "ASY_H",
    "PCN", "M_ms"
]

# Map common HRO fill values to NaN (keep '99' out of global na list; handle it only in time fields)
na_vals = [999999, 99999.9, 9999.99, 999.99, 99.99, 9999999., 9999999]

# 1) Read with regex separator (fixes FutureWarning) and robust line handling
df = pd.read_csv(
    INPUT_ASC,
    sep=r"\s+",
    header=None,
    names=cols,
    na_values=na_vals,
    engine="python",
    on_bad_lines="skip"  # skip malformed lines if any
)

# 2) Coerce time columns to numeric; mark '99' in Hour/Minute as missing
for c in ["Year", "DOY", "Hour", "Minute"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# In HRO minute files, 99 in Hour/Minute usually indicates missing time
df.loc[df["Hour"] == 99, "Hour"] = np.nan
df.loc[df["Minute"] == 99, "Minute"] = np.nan

# 3) Build datetime only for rows with complete time fields
time_ok = df[["Year", "DOY", "Hour", "Minute"]].notna().all(axis=1)
good = df.loc[time_ok].copy()

# Year-DOY to date, then add hour/minute
base = pd.to_datetime(
    good["Year"].astype(int).astype(str) + "-" +
    good["DOY"].astype(int).astype(str).str.zfill(3),
    format="%Y-%j"
)
good["datetime"] = base + pd.to_timedelta(good["Hour"].astype(int), unit="h") \
                          + pd.to_timedelta(good["Minute"].astype(int), unit="m")

# 4) Reorder columns (datetime first), save
out_cols = ["datetime"] + cols
good = good[out_cols]
good.to_csv(OUTPUT_CSV, index=False)

print(f"Saved CSV to: {OUTPUT_CSV}")
dropped = len(df) - len(good)
if dropped > 0:
    print(f"NOTE: dropped {dropped} rows without valid time fields (Hour/Minute/DOY/Year).")
