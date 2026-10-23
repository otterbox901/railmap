# Install

railmap is pure Python (3.10+) and needs no root to plan, print env vars or bind ranks. Run `railmap doctor` first on
any new host, container or job step: it checks everything below and prints a fix for each problem.

## venv / pip

```bash
python3 -m venv ~/.railmap && ~/.railmap/bin/pip install railmap
~/.railmap/bin/railmap doctor
```

Or `uv tool install railmap`, or `pipx install railmap`. From a checkout: `uv sync && uv run railmap doctor`.

- Ubuntu doesn't ship `venv` by default: `sudo apt-get install python3-venv` (22.04's Python 3.10 is fine).
- `mpirun` starts ranks over non-interactive ssh, which doesn't read `~/.bashrc`. Put railmap on the system `PATH`
  (`sudo ln -s ~/.railmap/bin/railmap /usr/local/bin/`) or launch it by full path.

System tools railmap uses when they are present:

| tool | package | used by |
|---|---|---|
| `numactl` | numactl | `exec` binds each rank's memory to its GPU's NUMA node |
| `rdma` | iproute2 | `observe` counts live QPs |
| `ib_write_bw` | perftest | `calibrate`; `gdr-matrix` needs a CUDA build |

## Container

```bash
docker run --rm --gpus all --device /dev/infiniband --ulimit memlock=-1 --network host \
  ghcr.io/otterbox901/railmap doctor
docker run --rm --gpus all --device /dev/infiniband ghcr.io/otterbox901/railmap plan --nodes 64
```

The image has railmap, rdma-core with the mlx5 provider, `rdma`, `numactl`, `ibv_devinfo` and perftest built with CUDA.
perftest loads `libcuda` only when asked to (`--use_cuda`), so the same image runs on CPU-only RDMA nodes. Build it
yourself with `docker build -t railmap .`.

| flag | why |
|---|---|
| `--gpus all` (Podman: `--device nvidia.com/gpu=all`) | NVML inside the container tells railmap which GPUs are yours |
| `--device /dev/infiniband` | RDMA verbs need the device nodes |
| `--ulimit memlock=-1` | RDMA memory registration needs pinned memory; Docker's default is 8 MiB |
| `--network host` | probes reach other nodes and see RoCE GIDs |
| `-v /opt/microsoft:/opt/microsoft:ro` | optional on Azure: railmap ships copies of the ND A100 v4 and ND H100 v5 files |

To use `railmap exec` inside your training image, install it there: see
[`examples/docker/Dockerfile.train`](../examples/docker/Dockerfile.train).

## What railmap sees in a container or job step

`/sys` shows the whole host even when a container or job was given part of it. railmap plans on the part you have:

- **GPUs:** the ones NVML can open (your allocation), numbered as CUDA numbers them under `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **NICs:** the ones with a `/dev/infiniband/uverbs*` node, when only some have one.
- **CPUs:** the cgroup's cpuset, so every rank on a node computes the same plan and gets a disjoint share of the CPUs it
  may run on. `--all-cpus` plans on the whole host instead. Snapshots and topology files for other hosts are never
  restricted.
- **Topology on Azure:** the host's `/opt/microsoft/*topo*.xml` if it is mounted, otherwise railmap's bundled copy
  when the GPU addresses match.
- **NUMA memory:** `exec` skips `numactl --membind` for a node outside the cgroup's `cpuset.mems`, with a warning.

The plan's notes say when any of this happened.

## Slurm

[`examples/slurm/`](../examples/slurm/):

- `railmap.sbatch`: `srun --cpu-bind=none railmap exec -- <cmd>`. Each rank binds itself; Slurm must not bind first.
- `railmap-mask.sbatch`: Slurm binds, using `railmap env --format slurm` for the `--cpu-bind=mask_cpu` list.
- `railmap-pyxis.sbatch`: Pyxis/Enroot, with `railmap doctor` from the railmap image and `exec` in your training image.

All three run `railmap doctor` once per node first, so a node without `/dev/infiniband` or a topology file fails before
the job starts.

## Kubernetes

[`examples/kubernetes/`](../examples/kubernetes/):

- `doctor-pod.yaml`: one pod that runs `railmap doctor` on a GPU node.
- `train-indexed-job.yaml`: a two-node `torchrun` job (Indexed Job + headless Service, no operator) with each rank under
  `railmap exec`. With Kubeflow's PyTorchJob or JobSet, use the same container command.

They expect the NVIDIA GPU Operator (`nvidia.com/gpu`) and Network Operator (an RDMA resource; the manifests use the
shared device plugin's default name `rdma/rdma_shared_device_a`). Add `IPC_LOCK` for RDMA. For CPUs near each GPU, run
the kubelet's static CPU manager with the Topology Manager and request whole CPUs with requests equal to limits;
`railmap doctor` shows which CPUs the pod received and warns when none are on a GPU's NUMA node.

## Troubleshooting

| `doctor` says | cause | fix |
|---|---|---|
| verbs devices: fail | the container has no `/dev/infiniband` | Docker `--device /dev/infiniband`; Kubernetes: request the RDMA resource |
| locked memory: warn | `ulimit -l` is small | Docker `--ulimit memlock=-1`; Slurm `PropagateResourceLimitsExcept=MEMLOCK` |
| topology: fail | virtualized PCI tree and no topology file | `--topo-file` / `NCCL_TOPO_FILE`, or mount the host's `/opt/microsoft` |
| GPUs: warn, NVML unavailable | no driver libraries in the container | Docker `--gpus all`; Kubernetes: request `nvidia.com/gpu` |
| GPUs: warn, under Podman with the GPU passed in | SELinux blocks `/dev/nvidia*` | `--security-opt=label=disable`, or `setsebool -P container_use_devices on` |
| CPUs: warn | the cgroup has no CPUs on a GPU's NUMA node | Kubernetes static CPU manager + Topology Manager; Slurm `--exclusive` |
| numactl: warn | numactl is missing on a multi-socket host | install numactl |
