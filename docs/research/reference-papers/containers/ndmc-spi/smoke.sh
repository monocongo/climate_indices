#!/bin/sh
# Run the NDMC SPI Generator under Mono and copy the results to OUTDIR.
#
#   smoke.sh OUTDIR                 the zip's bundled monthly sample (scales 1,2), plus the
#                                   zip's shipped sample output in OUTDIR/shipped for comparison
#   smoke.sh OUTDIR INPUT.csv       your own monthly series, scales 1,3. INPUT.csv is a station
#                                   name line, then `m/d/yyyy,value` rows, -99 for missing
#
# OUTDIR must be empty and the caller must not be root: the zip is third-party and
# untrusted, the program inherits your uid, and OUTDIR is mounted read-write into it.
#
# The program joins paths with a Windows backslash, so under Mono each output file is
# created in the working directory with a name like `o\NAME.csv`. The loop strips
# everything up to the last backslash on the way out.
set -eu
out=$(cd "$1" && pwd)
[ -z "$(ls -A "$out")" ] || { echo "OUTDIR must be empty: $out" >&2; exit 1; }
[ "$(id -u)" -ne 0 ] || { echo "do not run as root: the container inherits your uid" >&2; exit 1; }
docker build -q -t ndmc-spi:mono "$(dirname "$0")" >/dev/null
if [ $# -ge 2 ]; then
  name=$(basename "$2")
  cp "$2" "$out/$name"
  input="/out/$name"; scales=1,3
else
  input=/opt/ndmc/sampleinput/commadelimited_monthly.csv; scales=1,2
fi
# Run as the caller so the bind mount is writable on native Linux; /work is a tmpfs because the
# image's /work belongs to the image user, and HOME points at a writable path for Mono.
docker run --rm --network none --user "$(id -u):$(id -g)" --tmpfs /work:mode=1777 -e HOME=/tmp \
  -v "$out":/out -e INPUT="$input" -e SCALES="$scales" \
  --entrypoint sh ndmc-spi:mono -ec '
  mkdir -p /work/o && cd /work
  mono /opt/ndmc/executable/64bit/SPIGenerator.exe -i "$INPUT" -o /work/o \
    -s "$SCALES" -t monthly -f -p -l comma
  [ ! -e /work/exceptions.txt ] || { cat -v /work/exceptions.txt; exit 1; }
  for f in /work/o\\*; do cp "$f" "/out/${f##*\\}"; done
  mkdir -p /out/shipped && cp /opt/ndmc/sampleoutput/* /out/shipped/
'
