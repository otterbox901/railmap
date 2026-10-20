#!/usr/bin/env bash
# Runs on node0: topology capture, GDR matrix, QP-knee calibration, QP-inflation sweep and placement validation.
# Usage: remote-campaign.sh <out-dir> <node0-private-ip> <node1-private-ip>
set -o pipefail -o errexit -o nounset
out=$1 n0=$2 n1=$3
gpus=8
topo=/opt/microsoft/ndv5-topo.xml
mkdir -p "$out" && cd "$out"
nccl_tests=$(cat ~/.railmap-nccl-tests)
# shellcheck disable=SC1090
source /opt/hpcx-*/hpcx-init.sh && hpcx_load

echo "== 1. topology"
for h in "$n0" "$n1"; do
  ssh "$h" railmap show --json >"show-$h.json"
  ssh "$h" nvidia-smi topo -m >"nvidia-smi-topo-$h.txt"
  ssh "$h" ibv_devinfo -v >"ibv_devinfo-$h.txt" || true
done
railmap show | tee show.txt
railmap plan -o plan.json | tee plan.txt
hca0=$(python3 -c 'import json; print(json.load(open("plan.json"))["ranks"][0]["nics"][0].split(":")[0])')

echo "== 2. GPUDirect RDMA matrix (node0 GPU memory -> node1 host memory, same NIC index)"
railmap gdr-matrix --remote "$n1" --connect-to "$n1" -o gdr.json | tee gdr.txt

echo "== 3. QP knee on $hca0"
railmap calibrate --hca "$hca0" --server "$n1" --connect-to "$n1" -o profile.json | tee calibrate.txt

echo "== 4. QP inflation: 2 nodes, NCCL_IB_QPS_PER_CONNECTION raised to emulate larger jobs"
mpirun_base="mpirun -np $((2 * gpus)) -H $n0:$gpus,$n1:$gpus --bind-to none -x PATH -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0"
for q in 1 4 16 64 128; do
  # shellcheck disable=SC2086
  railmap observe -o "obs-q$q.json" -- $mpirun_base -x NCCL_IB_QPS_PER_CONNECTION=$q \
    railmap exec -- "$nccl_tests/alltoall_perf" -b 8 -e 1M -f 2 -g 1 >"a2a-q$q.txt" 2>&1
  railmap plan --nodes 2 --qps-per-conn "$q" --profile profile.json --json >"model-q$q.json"
done

echo "== 5. placement validation"
railmap validate --launcher "$mpirun_base" --nccl-tests-dir "$nccl_tests" --topo-file "$topo" -o validate.json \
  | tee validate.txt

echo "campaign finished: $out"
