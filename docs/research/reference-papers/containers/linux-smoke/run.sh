#!/bin/sh
# Run every smoke call, then compare with expected.tsv (the macOS arm64 values) at each value's recorded decimals.
set -u
V=/opt/venvs
echo "arch: $(uname -m)  python: $($V/xclim/bin/python -V)  $(R --version | head -1)"
$V/xclim/bin/python /smoke/inputs.py || exit 1
fail=0  # any comparator process failure fails the run, even if its rows match
{
  for pkg in xclim spei standard_precip pyet pyeto; do
    $V/$pkg/bin/python /smoke/smoke_py.py "$pkg" || { echo "ERROR	$pkg	script failed" >&2; fail=1; }
  done
  SMOKE_WORK=/work Rscript /smoke/smoke_r.R || { echo "ERROR	R	script failed" >&2; fail=1; }
} > /work/linux.tsv
echo "--- resolved versions"
for pkg in xclim spei standard_precip pyet; do
  freeze=$(uv pip freeze --python "$V/$pkg/bin/python") \
    || { echo "version capture failed: $pkg" >&2; exit 1; }
  echo "$pkg: $(printf '%s' "$freeze" | tr '\n' ' ')"
done
Rscript -e 'ip <- installed.packages(lib.loc = "/opt/rlib")[, "Version"]; cat(paste0(names(ip), "=", ip), fill = 100)' \
  || { echo "version capture failed: R" >&2; exit 1; }
echo "--- comparison (value | macOS expected | Linux)"
$V/xclim/bin/python - <<'PY'
import csv
exp = {r[0]: r[1] for r in csv.reader(open("/smoke/expected.tsv"), delimiter="\t")}
got = {}
structural = 0
if not exp:
    structural += 1
    print("DIFF expected.tsv is empty")
for r in csv.reader(open("/work/linux.tsv"), delimiter="\t"):
    if len(r) != 2:
        structural += 1
        print(f"DIFF unreadable row {r!r}")
        continue
    if r[0] in got:
        structural += 1
        print(f"DIFF duplicate label {r[0]}")
        continue
    got[r[0]] = r[1]
missing = 0
for k in sorted(exp.keys() - got.keys()):
    structural += 1
    missing += 1
    print(f"DIFF {k:34s} expected, not produced")
for k in sorted(got.keys() - exp.keys()):
    structural += 1
    print(f"DIFF {k:34s} produced, not in expected.tsv")
bad = 0
for k, e in exp.items():
    if k not in got:
        continue
    g = got[k]
    try:  # compare at the decimals the macOS survey recorded (six for most rows)
        dec = len(e.split(".")[1]) if "." in e else 0
        same = e == g == "NA" or (e != "NA" and g != "NA" and round(float(g), dec) == float(e))
    except ValueError:
        same = False
    bad += not same
    print(f"{'ok  ' if same else 'DIFF'} {k:34s} {e:>12s} {g:>20s}")
print(f"{len(exp) - bad - missing}/{len(exp)} reproduced, {structural} row error(s)")
raise SystemExit(bad > 0 or structural > 0)
PY
rc=$?
[ "$fail" -eq 0 ] || { echo "a comparator process failed: run fails" >&2; exit 1; }
exit "$rc"
