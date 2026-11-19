# NDMC SPI Generator under Mono, for the REFIMPL-03 comparator survey (issue 1605).
# The program is a Windows .NET 4.0 console application; Docker Desktop cannot run Windows
# containers, so this runs the shipped managed executable under Mono on Linux.
# The zip is third-party and untrusted: it is fetched with a pinned checksum and only ever
# executed inside the container (run it with --network none).
#
# Two stages: Debian's own Mono packages have no Microsoft.VisualBasic runtime (the
# program is VB.NET), and the Mono Project image has it but no unzip.
# debian:bookworm-slim; pinned by digest below (SonarCloud docker:S8431).
FROM debian@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc AS fetch
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip ca-certificates \
 && apt-get clean
# Served from drought.unl.edu; Last-Modified 2021-11-11, 5,071,776 bytes.
ADD --checksum=sha256:ce8f37bc00bf5309cdbedff2e788f698b1f30b2794e17b0e7aaab5c337efb4bf \
    https://drought.unl.edu/archive/Programs/SPI/SPIgenerator.zip /src/SPIgenerator.zip
RUN mkdir /opt/ndmc && cd /opt/ndmc && unzip -q /src/SPIgenerator.zip

# mono:6.12; pinned by digest below (SonarCloud docker:S8431).
FROM mono@sha256:34d816779b1248b5cfd095770b64ecbaf1798e2aca693a91c11a018dce9c7ad5
COPY --from=fetch /opt/ndmc /opt/ndmc
# Run unprivileged (SonarCloud docker:S6471). The program writes `o\NAME.csv` into its
# working directory, which this user owns; --create-home gives Mono a writable HOME.
RUN useradd --create-home --shell /bin/bash ndmc \
 && mkdir -p /work && chown ndmc:ndmc /work
USER ndmc
WORKDIR /work
ENTRYPOINT ["mono", "/opt/ndmc/executable/64bit/SPIGenerator.exe"]
