# How it works

```
railmap/
  topology/   sysfs + NVML discovery, NCCL XML parser, source selection (XML on virtualized VMs), cgroup cpuset
    vendor/   Azure ND A100 v4 / ND H100 v5 topology files (MIT, from Azure/azhpc-images)
  placement/  min-cost GPU->NIC matching, CPU split, QP-count model, budget against NIC profiles
  emit/       env vars, srun mask, exec wrapper, tables
  probe/      observe (rdma res show qp), calibrate (perftest -q sweep), GDR matrix, nccl-tests validation
  profiles/   NIC profiles; prior.json until something is measured
  ui/         `railmap ui`: stdlib HTTP server and a single self-contained page (no build step, no CDN)
  doctor.py   `railmap doctor`: environment checks, each with a fix
```

- **Path classes** use the `nvidia-smi topo -m` legend. They come from each device's PCI ancestor chain.
  - sysfs shows a PCIe switch as two bridge levels (upstream + downstream port), while NCCL XML shows it as one.
  - `NODE` vs `SYS` comes from NUMA affinity.
- **NIC assignment** uses a cost matrix: PIX 0, PXB 1, PHB 3, NODE 5, SYS 20.
  - Each NIC column is repeated `ceil(GPUs/NICs)` times so load stays balanced.
  - Ties break toward PCI order, so every node of a SKU gets the same, rail-aligned plan.
  - If a node has more NICs than GPUs, the spare NICs are shared out as secondary NICs.
- **What this process was given.** sysfs shows the whole host inside a container or job step, so discovery then keeps
  the GPUs NVML can open, the NICs with a verbs device node and the cgroup's CPUs. It reads the cgroup cpuset rather
  than per-process affinity: a launcher that pins ranks gives each a different affinity, but the same cgroup, so every
  rank still computes the same plan.
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
- Fake cgroup v1/v2 cpusets, `Mems_allowed_list` and `/dev/infiniband` nodes, for containers and job steps.
