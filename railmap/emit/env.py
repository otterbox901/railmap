"""Turn a plan into per-rank environment variables and launcher flags."""

from __future__ import annotations

from railmap.placement.assign import Plan, nic_names_known
from railmap.topology.model import format_cpulist


def nvshmem_pe_mapping(plan: Plan) -> str:
    """NVSHMEM_HCA_PE_MAPPING entries (hca:port:count), run-length encoded in local-rank order."""
    runs: list[list] = []
    for r in sorted(plan.ranks, key=lambda r: r.local_rank):
        if not r.nics:
            continue
        if runs and runs[-1][0] == r.nics[0]:
            runs[-1][1] += 1
        else:
            runs.append([r.nics[0], 1])
    return ",".join(f"{label}:{count}" for label, count in runs)


def rank_env(plan: Plan, local_rank: int, isolate_gpu: bool = False) -> dict[str, str]:
    r = plan.rank(local_rank)
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "RAILMAP_GPU": str(r.gpu),
        "RAILMAP_NIC": r.nics[0] if r.nics else "",
        "RAILMAP_CPUS": format_cpulist(r.cpus),
        "RAILMAP_NUMA": str(r.numa),
    }
    if isolate_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(r.gpu)
    if plan.topo_file:
        env["NCCL_TOPO_FILE"] = plan.topo_file
    if plan.hcas and r.nics and nic_names_known(plan.hcas):
        # The full set lets NCCL route through PXN; the "worst" variant pins one far NIC on purpose.
        hcas = [r.nics[0]] if plan.variant == "worst" else plan.hcas
        env["NCCL_IB_HCA"] = "=" + ",".join(hcas)
        env["UCX_NET_DEVICES"] = ",".join(r.nics)
        env["NVSHMEM_ENABLE_NIC_PE_MAPPING"] = "1"
        env["NVSHMEM_HCA_PE_MAPPING"] = nvshmem_pe_mapping(plan)
    return env


def shell_exports(env: dict[str, str]) -> str:
    return "\n".join(f"export {k}={_quote(v)}" for k, v in env.items())


def _quote(v: str) -> str:
    if v and all(c.isalnum() or c in "-_.,:/=" for c in v):
        return v
    return "'" + v.replace("'", "'\\''") + "'"


def slurm_cpu_bind(plan: Plan) -> str:
    """srun flag binding each local rank to its plan CPUs."""
    masks = []
    for r in sorted(plan.ranks, key=lambda r: r.local_rank):
        masks.append(hex(sum(1 << c for c in r.cpus)))
    return "--cpu-bind=mask_cpu:" + ",".join(masks)
