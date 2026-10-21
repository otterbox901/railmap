# Usage

```bash
uv sync
uv run railmap show                         # GPU x NIC path classes, NICs, NUMA nodes
uv run railmap plan                         # per-rank GPU, NIC, CPUs, NUMA node
uv run railmap plan --nodes 128             # ...plus the QP-per-NIC budget for a 128-node all-to-all
uv run railmap env --local-rank 3           # env vars for one rank
uv run railmap env --local-rank 0 --format slurm   # srun --cpu-bind mask for all ranks
torchrun --nproc-per-node 8 --no-python railmap exec -- python train.py   # bind each rank, then exec
mpirun -np 16 -H a:8,b:8 --bind-to none railmap exec -- alltoall_perf -b 8 -e 1G
```

## Topology sources

On an Azure HPC image, railmap uses `/opt/microsoft/*topo*.xml` automatically when it detects the virtualized PCI
tree. Pass `--topo-file` to use another NCCL topology XML (`NCCL_TOPO_FILE` is also honoured). To plan offline for
another machine, use `show --save topo.json` there, then `plan --snapshot topo.json` here.

## Job options

`plan --nodes N` options describe the job:

| option | meaning |
|---|---|
| `--pattern alltoall` / `allreduce` | traffic pattern |
| `--stack nccl` / `nvshmem` | which library opens the QPs |
| `--channels-per-peer`, `--qps-per-conn` | NCCL network channels per peer, `NCCL_IB_QPS_PER_CONNECTION` |
| `--rc-per-peer`, `--dc` | NVSHMEM IBGDA RC QPs per peer (DeepEP's `num_qps_per_rank`), or DC transport |
| `--rail-only` | hierarchical all-to-all: RDMA to same-rail peers, NVLink forwarding |
| `--profile` | NIC profile from [`railmap calibrate`](measurements.md#probes) (default: shipped profiles, else the prior) |

## What `exec` sets

| variable | value |
|---|---|
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID`, so CUDA indices match the plan |
| `NCCL_TOPO_FILE` | when a topology file was used |
| `NCCL_IB_HCA` | `=`-prefixed exact list of placement NICs (the full set, so PXN keeps working) |
| `UCX_NET_DEVICES` | the rank's NIC(s) |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING`, `NVSHMEM_HCA_PE_MAPPING` | each PE's NIC |
| CPU affinity | the rank's cores, from the GPU's NUMA node, SMT siblings kept together |
| `numactl --membind` | the rank's NUMA node, on multi-socket hosts |

`CUDA_VISIBLE_DEVICES` is set only with `--isolate-gpu`. Front-end NICs are left out of placement:
`mlx5_an0` on Azure, and in general any port slower than half the fastest.
