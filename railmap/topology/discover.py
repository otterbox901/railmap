"""Choose a topology source: snapshot > explicit/env topology file > vendor file on a flattened VM > live sysfs."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from railmap.topology import nccl_xml, nvml, sysfs
from railmap.topology.model import Nic, Topology

VENDOR_TOPO_GLOB = "opt/microsoft/*topo*.xml"  # Azure HPC images ship NCCL topology files here


def _match_vendor_file(live: Topology, root: Path) -> Path | None:
    want = {g.bdf for g in live.gpus}
    for candidate in sorted(root.glob(VENDOR_TOPO_GLOB)):
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


def discover(
    topo_file: Path | None = None,
    snapshot: Path | None = None,
    root: Path = Path("/"),
    env: Mapping[str, str] = os.environ,
) -> Topology:
    if snapshot:
        return Topology.load(snapshot)
    live = sysfs.scan(root)
    if root == Path("/"):
        info = nvml.gpu_info()
        for g in live.gpus:
            g.name, g.uuid = info.get(g.bdf, ("", ""))

    xml_path = topo_file or (Path(env["NCCL_TOPO_FILE"]) if env.get("NCCL_TOPO_FILE") else None)
    if xml_path is None and live.flattened:
        xml_path = _match_vendor_file(live, root)
        if xml_path:
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
