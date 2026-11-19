# NCL (dim_spi_n) from conda-forge, for the REFIMPL-03 comparator survey (issue 1605).
# conda-forge ships NCL only for linux-64, osx-64, and osx-arm64, and its osx-arm64 build
# contains no `ncl` interpreter, so this pins the linux-64 build and runs it under amd64
# emulation on Apple silicon.
FROM --platform=linux/amd64 mambaorg/micromamba:2.0.5-debian12-slim
# Latest linux-64 build as of 2026-11-18 (uploaded 2025-06-03).
# sha256 276f9f148e72904e6154ceec17927f97924d9ecf9c7fd45ba84a92224e94578a
RUN micromamba install -y -n base -c conda-forge ncl=6.6.2=h7a63124_57 \
 && micromamba clean -afy
ENV NCARG_ROOT=/opt/conda
WORKDIR /work
# The base image already runs as the unprivileged `mambauser`; declare it so the container
# user is explicit (SonarCloud docker:S6471). /work already belongs to that user.
USER mambauser
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "ncl"]
