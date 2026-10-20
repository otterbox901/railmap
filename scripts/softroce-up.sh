#!/usr/bin/env bash
# Create a local Soft-RoCE device (rxe0 on a dummy netdev) to exercise railmap's probes without RDMA hardware.
# Run with sudo. Throughput numbers from rxe are meaningless; this only checks code paths.
set -o pipefail -o errexit -o nounset
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
command -v ib_write_bw >/dev/null && command -v numactl >/dev/null || dnf install -y -q perftest numactl
modprobe rdma_rxe
ip link show rmtest0 >/dev/null 2>&1 || ip link add rmtest0 type dummy
ip link set rmtest0 up
ip addr replace 10.99.0.1/24 dev rmtest0
rdma link show rxe0/1 >/dev/null 2>&1 || rdma link add rxe0 type rxe netdev rmtest0
rdma link show rxe0/1
