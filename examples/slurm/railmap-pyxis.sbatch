#!/usr/bin/env bash
# Pyxis/Enroot: check each node with the railmap image, then run your training image (which has railmap installed;
# see examples/docker/Dockerfile.train). Pyxis passes GPUs and /dev/infiniband through on its own.
#SBATCH --job-name=railmap-pyxis
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=railmap-pyxis-%j.out
set -o pipefail -o errexit -o nounset

RAILMAP_IMAGE=${RAILMAP_IMAGE:-ghcr.io#otterbox901/railmap:latest}
TRAIN_IMAGE=${TRAIN_IMAGE:?set TRAIN_IMAGE to your training image (registry#name:tag)}
CMD=${CMD:-"python train.py"}

# Mounting the host's /opt/microsoft is optional on Azure: railmap ships copies of the ND v4/v5 topology files.
mounts=()
[[ -d /opt/microsoft ]] && mounts=(--container-mounts=/opt/microsoft:/opt/microsoft:ro)

srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  --container-image="$RAILMAP_IMAGE" "${mounts[@]}" railmap doctor
# shellcheck disable=SC2086  # CMD is a command line
srun --cpu-bind=none --mpi=pmix --container-image="$TRAIN_IMAGE" "${mounts[@]}" railmap exec -- $CMD
