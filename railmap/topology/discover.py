"""Choose a topology source: snapshot > explicit/env topology file > vendor file on a flattened VM > live sysfs.

On this host's topology (not a snapshot or a foreign file), GPUs, NICs and CPUs are then limited to the ones this
process was given: a container or job step sees the whole host in sysfs but may use only part of it.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from railmap.topology import cpuset, nccl_xml, nvml, sysfs
from railmap.topology.model import Nic, Topology

VENDOR_TOPO_GLOB = "opt/microsoft/*topo*.xml"  # Azure HPC images ship NCCL topology files here
BUNDLED_TOPO_DIR = Path(__file__).resolve().parent / "vendor"  # copies of those, for containers and other images


def _match_vendor_file(live: Topology, root: Path, bundled_dir: Path | None) -> Path | None:
    want = {g.bdf for g in live.gpus}
    candidates = sorted(root.glob(VENDOR_TOPO_GLOB))
    if bundled_dir:
        candidates += sorted(bundled_dir.glob("*.xml"))
    for candidate in candidates:
        try:
            if want and {g.bdf for g in nccl_xml.parse(candidate).gpus} == want:
                return candidate
        except (ET.ParseError, OSError):
            continue  # not a readable topology file
    return None


def merge(xml: Topology, live: Topology) -> Topology:
    """Keep the XML's structure (paths, NUMA) and fill names and port attributes from the live host by PCI address."""
    live_gpus = {g.bdf: g for g in live.gpus}
    if set(live_gpus) != {g.bdf for g in xml.gpus}:
        xml.notes.append("topology file does not match this host's GPUs; planning offline from the file alone")
        return xml
    for g in xml.gpus:
        g.name, g.uuid = live_gpus[g.bdf].name, live_gpus[g.bdf].uuid

    live_nics: dict[str, list[Nic]] = {}
    for n in live.nics:
        if n.bdf:
            live_nics.setdefault(n.bdf, []).append(n)
    nics: list[Nic] = []
    for n in xml.nics:
        matches = live_nics.pop(n.bdf, []) if n.bdf else []
        if not matches:
            nics.append(n)
            continue
        for ln in matches:
            nics.append(
                Nic(
                    name=ln.name,
                    bdf=n.bdf,
                    numa=n.numa,
                    pci_path=n.pci_path,
                    port=ln.port,
                    link_layer=ln.link_layer,
                    rate_gbps=ln.rate_gbps,
                    state=ln.state,
                    board_id=ln.board_id,
                    fw_ver=ln.fw_ver,
                    netdevs=ln.netdevs,
                )
            )
    for leftovers in live_nics.values():
        for ln in leftovers:
            xml.notes.append(f"ignoring {ln.label}: not in the topology file")
    xml.nics = nics
    xml.smt_siblings = live.smt_siblings
    xml.flattened = live.flattened
    xml.source = f"{xml.source}+sysfs"
    return xml


def restrict_cpus(topo: Topology, allowed: set[int], why: str) -> None:
    """Drop CPUs outside `allowed` from every NUMA node, so the plan only hands out CPUs ranks can run on."""
    total = sum(len(n.cpus) for n in topo.numa_nodes)
    for n in topo.numa_nodes:
        n.cpus = [c for c in n.cpus if c in allowed]
    kept = sum(len(n.cpus) for n in topo.numa_nodes)
    if kept < total:
        topo.notes.append(f"planning on {kept} of {total} CPUs allowed by the {why} (--all-cpus to plan on all)")


def keep_usable(topo: Topology, nvml_bdfs: set[str], verbs: dict[str, bool]) -> None:
    """Drop GPUs NVML can't open and NICs without a verbs device node, when only some are missing.

    When none are usable (no NVML, no /dev/infiniband), keep everything: that is a host without the driver or a
    container without the devices, and `railmap doctor` reports it; planning still works for inspection.
    """
    gpus = [g for g in topo.gpus if g.bdf in nvml_bdfs]
    if gpus and len(gpus) < len(topo.gpus):
        topo.notes.append(
            f"using the {len(gpus)} of {len(topo.gpus)} GPUs visible to NVML here: {', '.join(g.bdf for g in gpus)}"
        )
        for i, g in enumerate(gpus):
            g.index = i  # CUDA numbers only the visible GPUs, in PCI order
        topo.gpus = gpus
    opened = {name for name, ok in verbs.items() if ok}
    if opened and len(opened) < len(verbs):
        dropped = [n.label for n in topo.nics if n.name in verbs and n.name not in opened]
        topo.nics = [n for n in topo.nics if n.name not in verbs or n.name in opened]
        if dropped:
            topo.notes.append(f"not usable here (no /dev/infiniband node): {', '.join(dropped)}")


def describes_host(topo: Topology) -> bool:
    return topo.source == "sysfs" or topo.source.endswith("+sysfs")


def discover(
    topo_file: Path | None = None,
    snapshot: Path | None = None,
    root: Path = Path("/"),
    env: Mapping[str, str] = os.environ,
    all_cpus: bool = False,
    bundled_dir: Path | None = BUNDLED_TOPO_DIR,
) -> Topology:
    if snapshot:
        return Topology.load(snapshot)
    topo = _discover_host(topo_file, root, env, bundled_dir)
    if not describes_host(topo):
        return topo
    keep_usable(topo, {g.bdf for g in topo.gpus if g.uuid}, sysfs.uverbs_devices(root))  # only NVML sets uuid
    if not all_cpus:
        allowed = cpuset.allowed_cpus(root)
        if allowed:
            restrict_cpus(topo, *allowed)
    return topo


def _discover_host(topo_file: Path | None, root: Path, env: Mapping[str, str], bundled_dir: Path | None) -> Topology:
    live = sysfs.scan(root)
    if root == Path("/"):
        info = nvml.gpu_info()
        for g in live.gpus:
            g.name, g.uuid = info.get(g.bdf, ("", ""))

    xml_path = topo_file or (Path(env["NCCL_TOPO_FILE"]) if env.get("NCCL_TOPO_FILE") else None)
    if xml_path is None and live.flattened:
        xml_path = _match_vendor_file(live, root, bundled_dir)
        if xml_path and bundled_dir and xml_path.parent == bundled_dir:
            live.notes.append(
                f"virtualized PCI tree detected and no vendor topology file on this host; using railmap's copy of "
                f"{xml_path.name}"
            )
        elif xml_path:
            live.notes.append(f"virtualized PCI tree detected; using vendor topology file {xml_path}")
    if xml_path:
        topo = merge(nccl_xml.parse(xml_path), live)
        topo.notes = live.notes + topo.notes
        return topo
    return live


def placement_nics(topo: Topology) -> list[Nic]:
    """NICs eligible for placement: active ports, minus slow front-end NICs when fast ones exist."""
    active = [n for n in topo.nics if n.active]
    rates = [n.rate_gbps for n in active if n.rate_gbps]
    if not rates:
        return active
    fastest = max(rates)
    keep = [n for n in active if n.rate_gbps is None or n.rate_gbps >= fastest / 2]
    return keep
