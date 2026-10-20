# railmap

Places each rank of a multi-GPU job on the right NIC, CPU cores and NUMA memory. Also warns when a collective job
will open more RDMA connections per NIC than the NIC handles well.

It targets three gaps in multi-GPU / multi-NIC scaling guidance:

1. **Placement.** Each GPU needs a nearby NIC (PIX/PXB), and cross-socket (SYS) paths hurt. Launchers, containers
   and `CUDA_DEVICE_ORDER` mistakes still put ranks on the wrong NIC or socket. On cloud VMs the guest PCI tree is
   virtualized, so `nvidia-smi topo -m` and sysfs can't show the real layout.
2. **Queue sharding.** No published guidance says how many QPs each GPU should get on a shared NIC, or how to
   spread them across several nearby NICs.
3. **Connection state.** All-to-all RDMA reportedly loses about 59% of a NIC's message rate at about 3,000
   connections, and nothing warns a job before it gets there.

## Findings

Pending the Azure ND H100 v5 campaign (`infra/azure/`). The campaign will measure four things:

- the GPUDirect RDMA bandwidth for every GPU×NIC pair, grouped by path class;
- the message-rate knee against QP count on ConnectX-7 NDR;
- how closely the QP-count model matches QPs counted on the hardware;
- nccl-tests bus bandwidth under no binding, railmap's placement and a deliberately bad placement.

Until then, the only connection-budget profile is a one-point prior built from the 59% / 3,000 figure. Below 3,000
QPs, the expected loss is reported as unknown.

## Usage

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

On an Azure HPC image, railmap uses `/opt/microsoft/*topo*.xml` automatically when it detects the virtualized PCI
tree. Pass `--topo-file` to use another NCCL topology XML (`NCCL_TOPO_FILE` is also honoured). To plan offline for
another machine, use `show --save topo.json` there, then `plan --snapshot topo.json` here.

`plan --nodes N` options describe the job:

| option | meaning |
|---|---|
| `--pattern alltoall` / `allreduce` | traffic pattern |
| `--stack nccl` / `nvshmem` | which library opens the QPs |
| `--channels-per-peer`, `--qps-per-conn` | NCCL network channels per peer, `NCCL_IB_QPS_PER_CONNECTION` |
| `--rc-per-peer`, `--dc` | NVSHMEM IBGDA RC QPs per peer (DeepEP's `num_qps_per_rank`), or DC transport |
| `--rail-only` | hierarchical all-to-all: RDMA to same-rail peers, NVLink forwarding |
| `--profile` | NIC profile from `railmap calibrate` (default: shipped profiles, else the prior) |

### What `exec` sets

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

### Probes

These need RDMA hardware, or Soft-RoCE for a code-path check (`sudo scripts/softroce-up.sh`):

```bash
railmap observe -- <command>      # peak QPs per NIC owned by the command's process tree
railmap observe --duration 60     # whole host minus baseline (for nodes where ranks were launched remotely)
railmap calibrate --hca mlx5_ib0 --server node1 -o profile.json   # perftest -q sweep, knee fit
railmap gdr-matrix --remote node1 # GPUDirect RDMA Gb/s for every GPU x NIC pair
railmap validate --launcher "mpirun -np 16 -H a:8,b:8 --bind-to none"   # nccl-tests: none/best/worst
```

## Azure campaign

```bash
infra/azure/up.sh          # 2 x Standard_ND96isr_H100_v5 in one VMSS; prints cost, asks to confirm
infra/azure/bootstrap.sh   # railmap, CUDA perftest, nccl-tests; node0 -> node1 ssh
infra/azure/campaign.sh    # runs everything, fetches results/, then deletes the resource group
```

Requirements:

- An ND H100 v5 quota of at least 192 vCPUs in the region (`LOCATION` in `infra/azure/config.env`).
- Check the price per VM-hour before running; `PRICE_PER_VM_HOUR` in the same file feeds the confirmation prompt.

`campaign.sh` deletes the resource group when it exits. Set `KEEP=1` to keep it, then run `infra/azure/down.sh`
yourself.

The campaign runs on InfiniBand, not Ethernet. Profiles are keyed by NIC `board_id` and link layer, so a
RoCE profile measured elsewhere slots in without code changes.

## How it works

```
railmap/
  topology/   sysfs + NVML discovery, NCCL XML parser, source selection (XML on virtualized VMs)
  placement/  min-cost GPU->NIC matching, CPU split, QP-count model, budget against NIC profiles
  emit/       env vars, srun mask, exec wrapper, tables
  probe/      observe (rdma res show qp), calibrate (perftest -q sweep), GDR matrix, nccl-tests validation
  profiles/   NIC profiles; prior.json until something is measured
```

- **Path classes** use the `nvidia-smi topo -m` legend. They come from each device's PCI ancestor chain.
  - sysfs shows a PCIe switch as two bridge levels (upstream + downstream port), while NCCL XML shows it as one.
  - `NODE` vs `SYS` comes from NUMA affinity.
- **NIC assignment** uses a cost matrix: PIX 0, PXB 1, PHB 3, NODE 5, SYS 20.
  - Each NIC column is repeated `ceil(GPUs/NICs)` times so load stays balanced.
  - Ties break toward PCI order, so every node of a SKU gets the same, rail-aligned plan.
  - If a node has more NICs than GPUs, the spare NICs are shared out as secondary NICs.
- **QP estimates.**
  - NCCL all-to-all: `peers x channels/peer x QPs/connection x 2` (one connection per direction).
  - NVSHMEM RC: `peers x RC/peer`.
  - NVSHMEM DC: a constant.
  - These model library internals; `observe` gives the real count, and the campaign checks the model against it.

## Tests

```bash
uv run pytest
```

Fixtures:

- Azure's published `ndv5-topo.xml`.
- Fake sysfs trees: a DGX-like nested-switch box, a dual-socket "trap" where index-order pairing would cross
  sockets, a 4-GPU/8-NIC box and a flattened Hyper-V guest with and without the vendor file.
- This laptop's snapshot.
