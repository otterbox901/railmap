#!/usr/bin/env bash
# Slurm does the CPU binding, from railmap's plan, so `--cpu-bind=verbose` and accounting show it; railmap exec still
# sets NCCL/UCX/NVSHMEM variables and NUMA memory binding. The mask is computed on the first node, so nodes must be the
# same SKU (as in any homogeneous partition).
#SBATCH --job-name=railmap-mask
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=railmap-mask-%j.out
set -o pipefail -o errexit -o nounset

CMD=${CMD:-"all_reduce_perf -b 8 -e 1G -f 2 -g 1"}

bind=$(railmap env --local-rank 0 --format slurm)   # --cpu-bind=mask_cpu:0x...,0x...
echo "railmap: $bind"
# shellcheck disable=SC2086  # CMD is a command line
srun "$bind" --mpi=pmix railmap exec -- $CMD
