"""Parse NCCL topology XML (NCCL_TOPO_FILE, NCCL_TOPO_DUMP_FILE output, Azure /opt/microsoft/*-topo.xml)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from railmap.topology.model import Gpu, Nic, NumaNode, Topology


def parse_affinity(mask: str) -> list[int]:
    """NCCL affinity masks are comma-separated 32-bit hex words, most significant first."""
    value = int(mask.replace(",", ""), 16)
    return [i for i in range(value.bit_length()) if value >> i & 1]


def _nic_names(el: ET.Element) -> list[tuple[str, int]]:
    return [(net.get("name", ""), int(net.get("port", "1"))) for net in el.iter("net") if net.get("name")]


PLACEHOLDER_PREFIX = "pci-"  # NIC name when the file has no <net name=...> and no live host fills it in


def parse(path: Path) -> Topology:
    path = path.resolve()
    root = ET.parse(path).getroot()
    numa_nodes: list[NumaNode] = []
    gpus: list[tuple[str, int, list[str]]] = []
    nics: list[Nic] = []

    def walk(el: ET.Element, numa: int, chain: list[str]) -> None:
        for child in el:
            if child.tag == "nic":  # NIC without a PCI parent (virtual)
                for name, port in _nic_names(child):
                    nics.append(Nic(name=name, bdf=None, numa=numa, pci_path=[], port=port))
                continue
            if child.tag != "pci":
                continue
            busid = child.get("busid", "").lower()
            cls = child.get("class", "")
            has_gpu = child.find("gpu") is not None
            nic_el = child.find("nic")
            if has_gpu or cls.startswith("0x03"):
                gpus.append((busid, numa, chain))
            elif nic_el is not None or cls.startswith("0x02"):
                names = _nic_names(nic_el) if nic_el is not None else []
                for name, port in names or [(f"{PLACEHOLDER_PREFIX}{busid}", 1)]:
                    nics.append(Nic(name=name, bdf=busid, numa=numa, pci_path=chain, port=port))
            else:  # bridge / PCIe switch
                walk(child, numa, chain + [busid])

    for cpu in root.iter("cpu"):
        numa = int(cpu.get("numaid", "-1"))
        cpus = parse_affinity(cpu.get("affinity", "0"))
        numa_nodes.append(NumaNode(id=numa, cpus=cpus))
        walk(cpu, numa, [f"cpu{numa}"])

    return Topology(
        source=f"xml:{path}",
        gpus=[Gpu(index=i, bdf=b, numa=n, pci_path=c) for i, (b, n, c) in enumerate(sorted(gpus))],
        nics=nics,
        numa_nodes=sorted(numa_nodes, key=lambda n: n.id),
        levels_per_switch=1,
        topo_file=str(path),
    )
