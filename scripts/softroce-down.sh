#!/usr/bin/env bash
# Remove what softroce-up.sh created. Run with sudo.
set -o pipefail -o errexit -o nounset
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
rdma link delete rxe0 2>/dev/null || true
ip link delete rmtest0 2>/dev/null || true
modprobe -r rdma_rxe 2>/dev/null || true
