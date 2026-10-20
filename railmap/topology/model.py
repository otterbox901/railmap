"""Node topology: GPUs, RDMA NICs, NUMA nodes and the PCI paths between them."""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class PathClass(enum.IntEnum):
    """GPU-to-NIC path classes, same legend as `nvidia-smi topo -m` (lower is closer)."""

    PIX = 1  # at most one PCIe switch
    PXB = 2  # several PCIe switches, no host bridge
    PHB = 3  # through a PCIe host bridge (the CPU)
    NODE = 4  # between host bridges of one NUMA node
    SYS = 5  # across the inter-socket link


@dataclass
class Gpu:
    index: int  # CUDA index under CUDA_DEVICE_ORDER=PCI_BUS_ID
    bdf: str
    numa: int
    pci_path: list[str]  # ancestors from the PCI root down to the parent bridge
    name: str = ""
    uuid: str = ""


@dataclass
class Nic:
    name: str
    bdf: str | None  # None for virtual devices such as Soft-RoCE
    numa: int
    pci_path: list[str]
    port: int = 1
    link_layer: str = ""  # "InfiniBand" or "Ethernet"; empty when unknown (topology file only)
    rate_gbps: float | None = None
    state: str = ""  # "ACTIVE", "DOWN", ...; empty when unknown
    board_id: str = ""
    fw_ver: str = ""
    netdevs: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state in ("", "ACTIVE")

    @property
    def label(self) -> str:
        return f"{self.name}:{self.port}"


@dataclass
class NumaNode:
    id: int
    cpus: list[int]


@dataclass
class Topology:
    source: str
    gpus: list[Gpu]
    nics: list[Nic]
    numa_nodes: list[NumaNode]
    # sysfs shows each PCIe switch as two bridge levels (upstream + downstream port); NCCL XML collapses it to one.
    levels_per_switch: int = 2
    smt_siblings: dict[int, list[int]] = field(default_factory=dict)
    topo_file: str | None = None
    flattened: bool = False
    notes: list[str] = field(default_factory=list)

    def path(self, gpu: Gpu, nic: Nic) -> PathClass:
        return classify(gpu.pci_path, nic.pci_path, gpu.numa, nic.numa, self.levels_per_switch)

    def warnings(self) -> list[str]:
        if self.flattened and not self.topo_file:
            msg = (
                "virtualized PCI tree (Hyper-V vPCI) and no topology file: path classes are not real; pass --topo-file"
            )
            return [msg]
        return []

    def cpus_of(self, numa: int) -> list[int]:
        for node in self.numa_nodes:
            if node.id == numa:
                return list(node.cpus)
        return sorted(c for node in self.numa_nodes for c in node.cpus)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["smt_siblings"] = {str(k): v for k, v in self.smt_siblings.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Topology:
        return cls(
            source=d["source"],
            gpus=[Gpu(**g) for g in d["gpus"]],
            nics=[Nic(**n) for n in d["nics"]],
            numa_nodes=[NumaNode(**n) for n in d["numa_nodes"]],
            levels_per_switch=d.get("levels_per_switch", 2),
            smt_siblings={int(k): v for k, v in d.get("smt_siblings", {}).items()},
            topo_file=d.get("topo_file"),
            flattened=d.get("flattened", False),
            notes=list(d.get("notes", [])),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Topology:
        topo = cls.from_dict(json.loads(path.read_text()))
        topo.source = f"snapshot:{path}"
        return topo


def classify(a: list[str], b: list[str], numa_a: int, numa_b: int, levels_per_switch: int = 2) -> PathClass:
    """Classify the PCI path between two devices from their ancestor chains."""
    if not a or not b or a[0] != b[0]:
        if numa_a < 0 or numa_b < 0 or numa_a == numa_b:
            return PathClass.NODE
        return PathClass.SYS
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    if common == 1:
        return PathClass.PHB
    switches = 1 + (len(a) - common) // levels_per_switch + (len(b) - common) // levels_per_switch
    return PathClass.PIX if switches <= 1 else PathClass.PXB


def parse_cpulist(text: str) -> list[int]:
    """Parse a kernel cpulist such as "0-3,8,10-11"."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def format_cpulist(cpus: list[int]) -> str:
    """Inverse of parse_cpulist, collapsing runs into ranges."""
    out: list[str] = []
    s = sorted(set(cpus))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append(str(s[i]) if i == j else f"{s[i]}-{s[j]}")
        i = j + 1
    return ",".join(out)
