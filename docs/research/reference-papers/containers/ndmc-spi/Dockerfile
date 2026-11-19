# NDMC SPI Generator under Mono, for the REFIMPL-03 comparator survey (issue 1605).
# The program is a Windows .NET 4.0 console application; Docker Desktop cannot run Windows
# containers, so this runs the shipped managed executable under Mono on Linux.
# The zip is third-party and untrusted: it is fetched with a pinned checksum and only ever
# executed inside the container (run it with --network none).
#
# Two stages: Debian's own Mono packages have no Microsoft.VisualBasic runtime (the
# program is VB.NET), and the Mono Project image has it but no unzip.
FROM debian:bookworm-slim AS fetch
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip ca-certificates \
 && apt-get clean
# Served from drought.unl.edu; Last-Modified 2021-11-11, 5,071,776 bytes.
ADD --checksum=sha256:ce8f37bc00bf5309cdbedff2e788f698b1f30b2794e17b0e7aaab5c337efb4bf \
    https://drought.unl.edu/archive/Programs/SPI/SPIgenerator.zip /src/SPIgenerator.zip
RUN mkdir /opt/ndmc && cd /opt/ndmc && unzip -q /src/SPIgenerator.zip

FROM mono:6.12
COPY --from=fetch /opt/ndmc /opt/ndmc
WORKDIR /opt/ndmc
ENTRYPOINT ["mono", "/opt/ndmc/executable/64bit/SPIGenerator.exe"]
