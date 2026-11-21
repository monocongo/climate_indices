#!/bin/sh
# Download URL to OUT and fail unless its sha256 matches. Usage: fetch.sh URL SHA256 OUT
set -eu
curl -fsSL --retry 5 --retry-all-errors -o "$3" "$1"
echo "$2  $3" | sha256sum --strict -c -
