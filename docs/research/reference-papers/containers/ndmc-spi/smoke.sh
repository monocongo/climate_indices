#!/bin/sh
# Run the NDMC SPI Generator under Mono and copy the results to OUTDIR.
#
#   smoke.sh OUTDIR                 the zip's bundled monthly sample (scales 1,2), plus the
#                                   zip's shipped sample output in OUTDIR/shipped for comparison
#   smoke.sh OUTDIR INPUT.csv       your own monthly series, scales 1,3. INPUT.csv is a station
#                                   name line, then `m/d/yyyy,value` rows, -99 for missing
#
# The program joins paths with a Windows backslash, so under Mono each output file is
# created in the working directory with a name like `o\NAME.csv`. The loop strips
# everything up to the last backslash on the way out.
set -eu
out=$(cd "$1" && pwd)
docker build -q -t ndmc-spi:mono "$(dirname "$0")" >/dev/null
if [ $# -ge 2 ]; then
  name=$(basename "$2")
  cp "$2" "$out/$name"
  input="/out/$name"; scales=1,3
else
  input=/opt/ndmc/sampleinput/commadelimited_monthly.csv; scales=1,2
fi
docker run --rm --network none -v "$out":/out -e INPUT="$input" -e SCALES="$scales" \
  --entrypoint sh ndmc-spi:mono -c '
  mkdir -p /work/o && cd /work
  mono /opt/ndmc/executable/64bit/SPIGenerator.exe -i "$INPUT" -o /work/o \
    -s "$SCALES" -t monthly -f -p -l comma
  [ ! -e /work/exceptions.txt ] || { cat /work/exceptions.txt; exit 1; }
  for f in /work/o\\*; do cp "$f" "/out/${f##*\\}"; done
  mkdir -p /out/shipped && cp /opt/ndmc/sampleoutput/* /out/shipped/
'
