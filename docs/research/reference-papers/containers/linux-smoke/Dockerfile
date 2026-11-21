# Linux confirmation of the Python and R comparators (follow-up to REFIMPL-03, issue 1605).
# Pins and sha256 values come from ../../comparators.lock.yaml; tests/test_comparators_lock.py
# checks that each download uses its own locked hash. The build fails if a downloaded sdist or
# CRAN tarball does not match its recorded sha256. Neither the Python transitive dependencies
# nor the CRAN ones are pinned, and the two images and the apt packages are tag-pinned only:
# the versions the image resolves are printed by run.sh and recorded in comparators.md.
#
#   docker build -t refimpl-linux-smoke docs/research/reference-papers/containers/linux-smoke
#   docker run --rm --network none -v "$PWD/tests/fixture:/fixture:ro" refimpl-linux-smoke
# The flags are not optional: the image cannot enforce them, and --network none keeps the
# unpinned install-time code away from the network.
#
# Add --platform linux/amd64 to build and run for the CI-parity check (emulated on Apple silicon).
FROM rocker/r-ver:4.6.2

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

COPY fetch.sh /usr/local/bin/fetch.sh

# Python: one Python 3.12.16 venv per package. PyETo is pinned by commit (not on PyPI).
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN set -eu; \
    base=https://files.pythonhosted.org/packages/source; \
    fetch() { local url=$1 hash=$2 name=$3; fetch.sh "$url" "$hash" "/tmp/$name"; }; \
    inst() { local name=$1 archive=$2; \
             uv venv --python 3.12.16 "/opt/venvs/$name"; \
             uv pip install --python "/opt/venvs/$name/bin/python" "/tmp/$archive"; }; \
    mkdir -p /opt/venvs; \
    fetch "$base/x/xclim/xclim-1.0.0.tar.gz" \
        e06477defa0ce9bb599e426ff129c3f729bceae856c38cccdd1b69aeccf1c3d6 xclim-1.0.0.tar.gz; \
    fetch "$base/s/spei/spei-0.8.3.tar.gz" \
        502214088d77e2fcbae515747bff411577b843b2a8ed4fb82e9bef3e30ecc82f spei-0.8.3.tar.gz; \
    fetch "$base/s/standard-precip/standard_precip-2.0.0.tar.gz" \
        6a1f65c093568248a093d9ef14e76d62dade6dee6064a381b9f044980f0ce323 standard_precip-2.0.0.tar.gz; \
    fetch "$base/p/pyet/pyet-1.5.0.tar.gz" \
        81bedbc592df50f83e917dd76b5d239516c150bae060c39c536169c57042888b pyet-1.5.0.tar.gz; \
    inst xclim xclim-1.0.0.tar.gz; \
    inst spei spei-0.8.3.tar.gz; \
    inst standard_precip standard_precip-2.0.0.tar.gz; \
    inst pyet pyet-1.5.0.tar.gz; \
    uv venv --python 3.12.16 /opt/venvs/pyeto; \
    uv pip install --only-binary :all: --python /opt/venvs/pyeto/bin/python \
        numpy==2.5.5 pandas==3.1.1; \
    git clone https://github.com/woodcrafty/PyETo /opt/PyETo; \
    git -C /opt/PyETo checkout --quiet 79ddd88d727c7ebdad0ce26e220528d2921b66ea; \
    test "$(git -C /opt/PyETo rev-parse HEAD)" = 79ddd88d727c7ebdad0ce26e220528d2921b66ea; \
    rm -f /tmp/*.tar.gz

# R: a private library. The CRAN tarballs are hash-checked, then installed with their dependencies.
# TAR=internal makes R unpack with its own untar: GNU tar fails to create nested directories
# (`Cannot mkdir: Function not implemented`) on the climatekit tarball under linux/amd64
# emulation on Apple silicon. Plain `ubuntu:26.04` reproduces it, so it is not a package fault.
ENV R_LIBS=/opt/rlib
RUN set -eu; mkdir -p /opt/rlib; \
    cran() { local file=$1 hash=$2; \
             fetch.sh "https://cran.r-project.org/src/contrib/$file" "$hash" "/tmp/$file"; }; \
    cran SPEI_1.8.1.tar.gz 56cfcf6c7b7dc34fd05aa9cca3ee29d3b0628c82bcde80d6dc1519586483c7f6; \
    cran SCI_1.0-3.tar.gz 1ec42a6c6ee7e238eca0e4e627cef286cd2d46eaca417c86f3cc27fb3aa4011a; \
    cran SEI_0.2.0.tar.gz 55ac099c9d0794e61c792c30dc340de655d05875fdd6e15c82b887c097900c93; \
    cran Evapotranspiration_1.16.tar.gz 2550a842196941f1e6c40be5e61ee480aab96d13b8828154980a7207844bbe55; \
    cran climatekit_0.2.3.tar.gz f3bdc9f8425c35c51174a011606d810083f4b87e81e2df13d7229b6e3e023531; \
    TAR=internal Rscript -e \
        'install.packages("remotes"); \
         for (f in Sys.glob("/tmp/*.tar.gz")) remotes::install_local(f, upgrade = "never", \
             dependencies = NA)'; \
    rm -f /tmp/*.tar.gz

COPY inputs.py smoke_py.py smoke_r.R expected.tsv run.sh /smoke/
RUN useradd --create-home smoke && mkdir /work && chown smoke /work
USER smoke
WORKDIR /work
CMD ["sh", "/smoke/run.sh"]
