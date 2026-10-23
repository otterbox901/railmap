#!/usr/bin/env bash
# Every rank binds itself: railmap plans the node from sysfs and the step's cgroup, then takes its slot by SLURM_LOCALID.
# Slurm must not bind first (--cpu-bind=none), or the plan's CPUs would be outside each task's affinity.
#
#   sbatch examples/slurm/railmap.sbatch                       # nccl-tests all_reduce_perf
#   CMD="python train.py" sbatch examples/slurm/railmap.sbatch # your program (reads SLURM_PROCID / SLURM_NTASKS)
#SBATCH --job-name=railmap
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=railmap-%j.out
set -o pipefail -o errexit -o nounset

CMD=${CMD:-"all_reduce_perf -b 8 -e 1G -f 2 -g 1"}

# One check per node first: a missing /dev/infiniband or a flattened PCI tree without a topology file fails here.
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 railmap doctor

# shellcheck disable=SC2086  # CMD is a command line
srun --cpu-bind=none --mpi=pmix railmap exec -- $CMD
