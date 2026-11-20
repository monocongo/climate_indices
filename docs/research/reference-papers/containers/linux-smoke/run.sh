#!/bin/sh
# Run every smoke call, then compare with expected.tsv (the macOS arm64 values) at six decimals.
set -u
V=/opt/venvs
echo "arch: $(uname -m)  python: $($V/xclim/bin/python -V)  $(R --version | head -1)"
$V/xclim/bin/python /smoke/inputs.py || exit 1
{
  for pkg in xclim spei standard_precip pyet pyeto; do
    $V/$pkg/bin/python /smoke/smoke_py.py "$pkg" || echo "ERROR	$pkg	script failed" >&2
  done
  Rscript /smoke/smoke_r.R || echo "ERROR	R	script failed" >&2
} > /work/linux.tsv
echo "--- resolved versions"
for pkg in xclim spei standard_precip pyet; do
  echo "$pkg: $(uv pip freeze --python $V/$pkg/bin/python | tr '\n' ' ')"
done
Rscript -e 'ip <- installed.packages(lib.loc = "/opt/rlib")[, "Version"]; cat(paste0(names(ip), "=", ip), fill = 100)'
echo "--- comparison (value | macOS expected | Linux)"
$V/xclim/bin/python - <<'PY'
import csv
exp = {r[0]: r[1] for r in csv.reader(open("/smoke/expected.tsv"), delimiter="\t")}
got = {}
for r in csv.reader(open("/work/linux.tsv"), delimiter="\t"):
    got[r[0]] = r[1]
bad = 0
for k, e in exp.items():
    g = got.get(k, "MISSING")
    try:  # compare at the decimals the macOS survey recorded (six for most rows)
        dec = len(e.split(".")[1]) if "." in e else 0
        same = e == g == "NA" or (e != "NA" and g != "NA" and round(float(g), dec) == float(e))
    except ValueError:
        same = False
    bad += not same
    print(f"{'ok  ' if same else 'DIFF'} {k:34s} {e:>12s} {g:>12s}")
print(f"{len(exp) - bad}/{len(exp)} reproduced")
raise SystemExit(bad > 0)
PY
