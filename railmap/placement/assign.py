"""Assign each local rank a GPU, its closest NIC(s), CPU cores and a NUMA memory node."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from railmap.topology.discover import placement_nics
from railmap.topology.model import Gpu, Nic, PathClass, Topology
from railmap.topology.nccl_xml import PLACEHOLDER_PREFIX

PATH_COST = {PathClass.PIX: 0.0, PathClass.PXB: 1.0, PathClass.PHB: 3.0, PathClass.NODE: 5.0, PathClass.SYS: 20.0}


@dataclass
class RankPlacement:
    local_rank: int
    gpu: int
    gpu_bdf: str
    nics: list[str]  # "name:port", primary first
    path: str | None  # PathClass name of the primary NIC
    numa: int
    cpus: list[int]


@dataclass
class Plan:
    topology_source: str
    topo_file: str | None
    variant: str
    ranks: list[RankPlacement]
    hcas: list[str]  # every placement NIC, for NCCL_IB_HCA
    link_layers: dict[str, str] = field(default_factory=dict)
    numa_count: int = 1
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    version: int = 1

    def rank(self, local_rank: int) -> RankPlacement:
        for r in self.ranks:
            if r.local_rank == local_rank:
                return r
        raise KeyError(f"local rank {local_rank} not in plan (plan has {len(self.ranks)} ranks)")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Plan:
        d = json.loads(path.read_text())
        d["ranks"] = [RankPlacement(**r) for r in d["ranks"]]
        return cls(**d)


def nic_names_known(nics: list[Nic] | list[str]) -> bool:
    names = [n if isinstance(n, str) else n.name for n in nics]
    return not any(n.startswith(PLACEHOLDER_PREFIX) for n in names)


def _match(cost: np.ndarray, capacity: int, maximize: bool = False) -> dict[int, int]:
    """Assign rows to columns, each column taking at most `capacity` rows. Returns row -> column."""
    cols = cost.shape[1]
    slots = np.repeat(cost, capacity, axis=1)
    # Fill every column's first slot before any second one, so load stays balanced.
    slots = slots + np.tile(np.arange(capacity) * 1e-3, cols)
    r, c = linear_sum_assignment(slots, maximize=maximize)
    return {int(i): int(j) // capacity for i, j in zip(r, c)}


def _cost_matrix(topo: Topology, rows: list, cols: list, row_is_gpu: bool) -> np.ndarray:
    m = np.zeros((len(rows), len(cols)))
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            gpu, nic = (a, b) if row_is_gpu else (b, a)
            # Tie-break toward proportional position so equal-cost choices stay in PCI order (rail-aligned).
            m[i, j] = PATH_COST[topo.path(gpu, nic)] + 1e-6 * abs(i / len(rows) - j / len(cols))
    return m


def _assign_nics(topo: Topology, gpus: list[Gpu], nics: list[Nic], worst: bool) -> dict[int, list[Nic]]:
    out: dict[int, list[Nic]] = {i: [] for i in range(len(gpus))}
    if not nics:
        return out
    primary = _match(_cost_matrix(topo, gpus, nics, True), math.ceil(len(gpus) / len(nics)), maximize=worst)
    for gi, ni in primary.items():
        out[gi].append(nics[ni])
    if not worst and len(nics) > len(gpus):
        used = set(primary.values())
        spare = [n for i, n in enumerate(nics) if i not in used]
        extra = _match(_cost_matrix(topo, spare, gpus, False), math.ceil(len(spare) / len(gpus)))
        for ni, gi in sorted(extra.items()):
            out[gi].append(spare[ni])
    return out


def _core_groups(topo: Topology, cpus: list[int]) -> list[list[int]]:
    """Group CPUs into physical cores (SMT siblings together), ordered by first CPU."""
    allowed = set(cpus)
    seen: set[int] = set()
    groups = []
    for c in sorted(cpus):
        if c in seen:
            continue
        group = sorted(s for s in topo.smt_siblings.get(c, [c]) if s in allowed) or [c]
        seen.update(group)
        groups.append(group)
    return groups


def _split(groups: list[list[int]], k: int) -> list[list[int]]:
    base, extra = divmod(len(groups), k)
    out, i = [], 0
    for r in range(k):
        n = base + (1 if r < extra else 0)
        out.append([c for g in groups[i : i + n] for c in g])
        i += n
    return out


def assign(topo: Topology, ranks: int | None = None, variant: str = "best") -> Plan:
    if variant not in ("best", "worst"):
        raise ValueError(f"unknown variant {variant!r}")
    if not topo.gpus:
        raise ValueError("no GPUs in topology")
    ranks = ranks or len(topo.gpus)
    if ranks > len(topo.gpus):
        raise ValueError(f"{ranks} ranks requested but only {len(topo.gpus)} GPUs")
    worst = variant == "worst"
    gpus = topo.gpus[:ranks]
    nics = placement_nics(topo)
    warnings = topo.warnings()
    notes = list(topo.notes)
    dropped = [n.label for n in topo.nics if n not in nics]
    if dropped:
        notes.append(f"not used for placement (inactive or slower front-end ports): {', '.join(dropped)}")
    if not nics:
        warnings.append("no usable RDMA NICs: plan covers CPU and NUMA binding only")
    elif not nic_names_known(nics):
        notes.append("NIC device names are unknown offline, so NIC env vars are omitted; plan on the target host")

    nic_map = _assign_nics(topo, gpus, nics, worst)

    numa_ids = [n.id for n in topo.numa_nodes]
    rank_numa = {}
    for i, g in enumerate(gpus):
        numa = g.numa
        if worst and len(numa_ids) > 1 and numa in numa_ids:
            numa = numa_ids[(numa_ids.index(numa) + 1) % len(numa_ids)]
        rank_numa[i] = numa
        if g.numa < 0 and len(numa_ids) > 1:
            warnings.append(f"GPU{g.index} has no NUMA affinity in the topology; binding to all CPUs")

    cpus: dict[int, list[int]] = {}
    for numa in sorted(set(rank_numa.values())):
        members = [i for i, n in rank_numa.items() if n == numa]
        groups = _core_groups(topo, topo.cpus_of(numa))
        if len(groups) < len(members):
            warnings.append(f"NUMA node {numa}: {len(members)} ranks share {len(groups)} cores")
            for i in members:
                cpus[i] = [c for g in groups for c in g]
            continue
        for i, chunk in zip(members, _split(groups, len(members))):
            cpus[i] = chunk

    placements = []
    for i, g in enumerate(gpus):
        assigned = nic_map[i]
        path = topo.path(g, assigned[0]) if assigned else None
        if path is PathClass.SYS and not worst:
            closer = any(topo.path(g, n) < PathClass.SYS for n in nics)
            why = "closer NICs are already taken" if closer else "no NIC shares its socket"
            warnings.append(f"GPU{g.index} -> {assigned[0].label} crosses sockets (SYS): {why}")
        elif path in (PathClass.PHB, PathClass.NODE) and not worst:
            warnings.append(
                f"GPU{g.index} -> {assigned[0].label} is {path.name}: GPUDirect RDMA goes through the CPU root "
                "complex and NCCL may fall back to staging through host memory (see NCCL_NET_GDR_LEVEL)"
            )
        placements.append(
            RankPlacement(
                local_rank=i,
                gpu=g.index,
                gpu_bdf=g.bdf,
                nics=[n.label for n in assigned],
                path=path.name if path else None,
                numa=rank_numa[i],
                cpus=cpus[i],
            )
        )

    return Plan(
        topology_source=topo.source,
        topo_file=topo.topo_file,
        variant=variant,
        ranks=placements,
        hcas=[n.label for n in nics],
        link_layers={n.label: n.link_layer for n in nics},
        numa_count=len(topo.numa_nodes),
        warnings=warnings,
        notes=notes,
    )
