BASE="https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
OUTDIR="./omni"
mkdir -p "$OUTDIR"
for y in {2011..2025}; do
  wget -c -t 5 --timeout=30 -P "$OUTDIR" "$BASE/omni2_${y}.dat"
done
