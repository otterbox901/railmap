# railmap with the tools its probes use: rdma-core (verbs + mlx5 provider), iproute2's `rdma`, numactl, and perftest
# built with CUDA support. perftest loads libcuda at run time, so the same binary works on CPU-only RDMA nodes, and the
# runtime image needs no CUDA base (and so no NVIDIA_REQUIRE_CUDA driver check).
#
#   docker build -t railmap .
#   docker run --rm --gpus all --device /dev/infiniband --ulimit memlock=-1 --network host railmap doctor

ARG CUDA_DEVEL_IMAGE=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04
ARG BASE_IMAGE=docker.io/library/ubuntu:24.04

FROM ${CUDA_DEVEL_IMAGE} AS perftest
# Same commit as infra/azure/node-setup.sh (26.07.8).
ARG PERFTEST_REF=a31f9ea9df720ee68f941e55f75fec31349fcd35
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git autoconf automake libtool make pkg-config \
      libibverbs-dev librdmacm-dev libibumad-dev libpci-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git init -q /src \
 && git -C /src fetch -q --depth 1 https://github.com/linux-rdma/perftest "$PERFTEST_REF" \
 && git -C /src checkout -q FETCH_HEAD
WORKDIR /src
RUN ./autogen.sh \
 && ./configure -q --prefix=/opt/perftest --disable-cudart CUDA_H_PATH=/usr/local/cuda/include/cuda.h \
 && make -s -j"$(nproc)" \
 && make -s install \
 && /opt/perftest/bin/ib_write_bw --help | grep -q use_cuda

FROM ${BASE_IMAGE} AS base
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
      libibverbs1 ibverbs-providers ibverbs-utils librdmacm1t64 libibumad3 libpci3 iproute2 numactl \
 && rm -rf /var/lib/apt/lists/*

FROM base AS app
COPY pyproject.toml README.md LICENSE /src/
COPY railmap /src/railmap
RUN python3 -m venv /opt/railmap \
 && /opt/railmap/bin/pip install --no-cache-dir --disable-pip-version-check /src

FROM base
ARG VERSION=dev
LABEL org.opencontainers.image.title="railmap" \
      org.opencontainers.image.description="GPU-NIC-CPU placement and RDMA connection-budget planner for multi-GPU nodes" \
      org.opencontainers.image.source="https://github.com/otterbox901/railmap" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"
COPY --from=perftest /opt/perftest/bin/ /usr/local/bin/
COPY --from=app /opt/railmap /opt/railmap
# GPUs come from the runtime (--gpus, the Kubernetes device plugin); only ask for the driver libraries NVML needs.
ENV PATH=/opt/railmap/bin:$PATH \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["railmap"]
CMD ["doctor"]
