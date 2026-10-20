"""Live topology discovery from /sys. Every function takes `root` so tests can point it at a fake tree."""

from __future__ import annotations

import os
import re
from pathlib import Path

from railmap.topology.model import Gpu, Nic, NumaNode, Topology, parse_cpulist

BDF_RE = re.compile(r"^[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
ROOT_RE = re.compile(r"^pci[0-9a-f]{4,8}:[0-9a-f]{2}$")
NVIDIA_VENDOR = "0x10de"
GPU_CLASSES = ("0x0300", "0x0302")  # VGA, 3D controller


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


def normalize_bdf(bdf: str) -> str:
    """Lower-case a PCI address and shorten an 8-digit domain (NVML style) to 4 digits."""
    bdf = bdf.strip().lower()
    domain, rest = bdf.split(":", 1)
    return f"{int(domain, 16):04x}:{rest}"


def pci_chain(dev_path: Path, sys_root: Path) -> tuple[list[str], bool]:
    """Ancestors of a resolved PCI device path, from the `pciDDDD:BB` root down to the parent bridge.

    Also returns whether the device hangs off Hyper-V's VMBus, i.e. the guest sees a flattened virtual PCI tree.
    """
    resolved = Path(os.path.realpath(dev_path))
    try:
        parts = resolved.relative_to(Path(os.path.realpath(sys_root / "devices"))).parts
    except ValueError:
        parts = resolved.parts
    vmbus = any(p.upper().startswith("VMBUS") for p in parts)
    for i, p in enumerate(parts):
        if ROOT_RE.match(p):
            return list(parts[i:-1]), vmbus
    return [], vmbus


def _numa(dev: Path, single_node: bool) -> int:
    n = int(_read(dev / "numa_node", "-1") or -1)
    return 0 if n < 0 and single_node else n


def scan_numa(root: Path) -> list[NumaNode]:
    node_dir = root / "sys/devices/system/node"
    nodes = []
    for d in sorted(node_dir.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        nodes.append(NumaNode(id=int(d.name[4:]), cpus=parse_cpulist(_read(d / "cpulist"))))
    if not nodes:
        online = _read(root / "sys/devices/system/cpu/online") or f"0-{(os.cpu_count() or 1) - 1}"
        nodes = [NumaNode(id=0, cpus=parse_cpulist(online))]
    return nodes


def scan_smt(root: Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for d in (root / "sys/devices/system/cpu").glob("cpu[0-9]*"):
        sib = _read(d / "topology/thread_siblings_list")
        if sib:
            out[int(d.name[3:])] = parse_cpulist(sib)
    return out


def scan_gpus(root: Path, single_node: bool) -> tuple[list[Gpu], bool]:
    sys_root = root / "sys"
    found: list[tuple[str, Path]] = []
    for dev in (sys_root / "bus/pci/devices").glob("*"):
        if _read(dev / "vendor") == NVIDIA_VENDOR and _read(dev / "class").startswith(GPU_CLASSES):
            found.append((normalize_bdf(dev.name), dev))
    gpus = []
    flattened = False
    for i, (bdf, dev) in enumerate(sorted(found)):
        chain, vmbus = pci_chain(dev, sys_root)
        flattened |= vmbus
        gpus.append(Gpu(index=i, bdf=bdf, numa=_numa(dev, single_node), pci_path=chain))
    return gpus, flattened


def _rate_gbps(text: str) -> float | None:
    m = re.match(r"\s*([\d.]+)\s*Gb/sec", text)
    return float(m.group(1)) if m else None


def scan_nics(root: Path, single_node: bool) -> tuple[list[Nic], bool]:
    sys_root = root / "sys"
    nics: list[Nic] = []
    flattened = False
    for ib in sorted((sys_root / "class/infiniband").glob("*")):
        dev = ib / "device"
        bdf: str | None = None
        chain: list[str] = []
        numa = 0 if single_node else -1
        if dev.exists() and BDF_RE.match(Path(os.path.realpath(dev)).name):
            bdf = normalize_bdf(Path(os.path.realpath(dev)).name)
            chain, vmbus = pci_chain(dev, sys_root)
            flattened |= vmbus
            numa = _numa(dev, single_node)
        netdevs = sorted(p.name for p in (dev / "net").glob("*")) if dev.exists() else []
        parent = _read(ib / "parent")  # Soft-RoCE names its netdev here
        if parent and parent not in netdevs:
            netdevs.append(parent)
        for port_dir in sorted((ib / "ports").glob("*"), key=lambda p: int(p.name)):
            state = _read(port_dir / "state")
            nics.append(
                Nic(
                    name=ib.name,
                    bdf=bdf,
                    numa=numa,
                    pci_path=chain,
                    port=int(port_dir.name),
                    link_layer=_read(port_dir / "link_layer"),
                    rate_gbps=_rate_gbps(_read(port_dir / "rate")),
                    state=state.split(":", 1)[-1].strip() if state else "UNKNOWN",
                    board_id=_read(ib / "board_id"),
                    fw_ver=_read(ib / "fw_ver"),
                    netdevs=netdevs,
                )
            )
    return nics, flattened


def scan(root: Path = Path("/")) -> Topology:
    numa = scan_numa(root)
    single = len(numa) == 1
    gpus, flat_g = scan_gpus(root, single)
    nics, flat_n = scan_nics(root, single)
    topo = Topology(
        source="sysfs",
        gpus=gpus,
        nics=nics,
        numa_nodes=numa,
        levels_per_switch=2,
        smt_siblings=scan_smt(root),
        flattened=flat_g or flat_n,
    )
    return topo
