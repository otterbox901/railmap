#!/usr/bin/env bash
# Runs on each VM: install railmap, and perftest (with CUDA) and nccl-tests if the image lacks them.
set -o pipefail -o errexit -o nounset

PERFTEST_REF=a31f9ea9df720ee68f941e55f75fec31349fcd35  # 26.07.8
NCCL_TESTS_REF=afd59abab774a6a4b492eb344bd0c845a0b7bb4d  # v2.21.1

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --force "$HOME/railmap"
# mpirun's non-interactive ssh shells don't read ~/.bashrc, so put railmap on the system PATH.
sudo ln -sf "$HOME/.local/bin/railmap" /usr/local/bin/railmap

command -v numactl >/dev/null || sudo apt-get install -y -q numactl

fetch() {  # fetch <url> <commit> <dir>
  rm -rf "$3" && git init -q "$3" && git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q FETCH_HEAD
}

if ! ib_write_bw --help 2>&1 | grep -q use_cuda; then
  fetch https://github.com/linux-rdma/perftest "$PERFTEST_REF" "$HOME/src/perftest"
  (cd "$HOME/src/perftest" && ./autogen.sh && ./configure -q --prefix=/usr/local \
    CUDA_H_PATH=/usr/local/cuda/include/cuda.h && make -s -j"$(nproc)" && sudo make -s install)
  hash -r
fi
ib_write_bw --help 2>&1 | grep -q use_cuda || { echo "perftest without CUDA support" >&2; exit 1; }

nccl_tests=""
for d in /opt/nccl-tests/build "$HOME/src/nccl-tests/build"; do
  [[ -x "$d/alltoall_perf" ]] && nccl_tests="$d" && break
done
if [[ -z "$nccl_tests" ]]; then
  # shellcheck disable=SC1090
  source /opt/hpcx-*/hpcx-init.sh && hpcx_load
  fetch https://github.com/NVIDIA/nccl-tests "$NCCL_TESTS_REF" "$HOME/src/nccl-tests"
  make -s -C "$HOME/src/nccl-tests" -j"$(nproc)" MPI=1 MPI_HOME="$HPCX_MPI_DIR" CUDA_HOME=/usr/local/cuda
  nccl_tests="$HOME/src/nccl-tests/build"
fi
echo "$nccl_tests" >"$HOME/.railmap-nccl-tests"
echo "$(hostname): railmap $(railmap --help >/dev/null && echo ok), nccl-tests in $nccl_tests"
