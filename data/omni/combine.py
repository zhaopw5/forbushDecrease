# -*- coding: utf-8 -*-
"""
Combine OMNI2 low_res_omni yearly files (2011-2025) into a single CSV.
- Input  : omni2_2011.dat ... omni2_2025.dat (put them in the same folder as this script)
- Output : omni2_2011_2025.csv (with header), first three time columns merged to 'datetime' (YYYY/MM/DD/HH)
- Notes  : Replace canonical OMNI2 fill values (99, 999, 999.9, 9999, 99999, 9999999, etc.) with NaN
"""

import os
import math
import glob
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ----------------------------
# Column names from omni2.text
# ----------------------------
COLUMNS = [
    # 1..55 (omni2.text WORD order)
    "Year", "DOY", "Hour", "BartelsRot",
    "ID_IMF", "ID_SW",
    "Npts_IMF", "Npts_SW",
    "B_avg_abs", "B_vec_mag", "B_vec_lat_gse", "B_vec_lon_gse",
    "Bx_gse", "By_gse", "Bz_gse", "By_gsm", "Bz_gsm",
    "sigma_Babs", "sigma_B", "sigma_Bx", "sigma_By", "sigma_Bz",
    "Tp", "Np", "Vsw", "V_lon", "V_lat",
    "alpha_to_proton", "P_dyn",
    "sigma_T", "sigma_N", "sigma_V", "sigma_V_phi", "sigma_V_theta", "sigma_alpha_ratio",
    "Ey_mVpm", "beta", "MA_alfven",
    "Kp", "Sunspot_R", "Dst", "AE",
    "pf_gt1MeV", "pf_gt2MeV", "pf_gt4MeV", "pf_gt10MeV", "pf_gt30MeV", "pf_gt60MeV",
    "pf_flag",
    "ap_index", "F10_7", "PCN", "AL", "AU",
    "Mach_ms"
]

# ----------------------------
# Missing-value patterns
# ----------------------------
# We will replace the canonical "all 9s" fills with NaN.
# This list covers OMNI2 integer and float fill encodings.
INT_MISSING = {99, 999, 9999, 99999, 999999}
FLOAT_MISSING = {
    9.0, 9.9, 9.99, 9.999, 99.0, 99.9, 99.99, 999.0, 999.9, 9999.0, 99999.0,
    999999.0, 9999999.0, 999999.99, 99999.99, 9999.99, 999.99
}
# Some fields use large fixed-width fills like 9999999. or 999999.99 etc.
# We'll also use type-aware rules below as a safety net.

def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace OMNI2 canonical fill values with NaN, column-wise."""
    for col in df.columns:
        if col in ("Year", "DOY", "Hour"):  # keep original time cols for conversion
            continue
        s = df[col]
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
            df[col] = s.where(~s.isin(INT_MISSING), np.nan)
        else:
            # try float conversion; if fails, leave as is
            try:
                vals = pd.to_numeric(s, errors="coerce")
                mask = vals.isin(list(FLOAT_MISSING)) | vals.isin(list(INT_MISSING))
                # also: anything >= 9e5 is very likely a fill in OMNI2 context
                mask |= vals >= 9e5
                df[col] = vals.mask(mask, np.nan)
            except Exception:
                pass
    return df

def to_datetime_str(row) -> str:
    """Convert Year + DOY + Hour to 'YYYY/MM/DD/HH' string."""
    y = int(row["Year"])
    doy = int(row["DOY"])
    hh = int(row["Hour"])
    base = datetime(y, 1, 1) + timedelta(days=doy - 1, hours=hh)
    return base.strftime("%Y/%m/%d/%H")

def read_omni2_year(path: str) -> pd.DataFrame:
    """
    Read a single omni2_YYYY.dat as whitespace-delimited table with 55 columns.
    OMNI2 fields are fixed-width but whitespace parsing works well and is robust across years.
    """
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        names=COLUMNS,
        na_filter=False,  # we'll replace fills ourselves
        engine="python"
    )
    # Replace fills with NaN
    df = replace_missing(df)
    # Build datetime string column
    dt = df.apply(to_datetime_str, axis=1)
    # Reorder columns: datetime + (others excluding original Year/DOY/Hour)
    others = [c for c in COLUMNS if c not in ("Year", "DOY", "Hour")]
    out = pd.concat([dt.rename("datetime"), df[others]], axis=1)
    return out

def main():
    years = list(range(2011, 2025 + 1))
    found_files = []
    for y in years:
        fn = f"omni2_{y}.dat"
        if os.path.isfile(fn):
            found_files.append(fn)
        else:
            print(f"[WARN] file not found, skip: {fn}")

    if not found_files:
        raise FileNotFoundError("No omni2_YYYY.dat files found in current directory.")

    frames = []
    for fn in sorted(found_files):
        print(f"[INFO] reading {fn}")
        df = read_omni2_year(fn)
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)

    # Optional: sort by datetime just in case
    big["__key"] = pd.to_datetime(big["datetime"], format="%Y/%m/%d/%H", utc=False)
    big.sort_values("__key", inplace=True)
    big.drop(columns="__key", inplace=True)

    out_csv = "omni2_2011_2025.csv"
    big.to_csv(out_csv, index=False)
    print(f"[OK] saved {out_csv} with {len(big):,} rows and {big.shape[1]} columns.")

if __name__ == "__main__":
    main()
