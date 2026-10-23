"""Build fake /sys trees (and /opt/microsoft) for topology tests."""

from __future__ import annotations

import shutil
from pathlib import Path

NVIDIA = "0x10de"
MELLANOX = "0x15b3"
GPU_CLASS = "0x030200"
IB_CLASS = "0x020700"
FIXTURES = Path(__file__).parent / "fixtures"
VMBUS_PREFIX = ["LNXSYSTM:00", "LNXSYBUS:00", "ACPI0004:00", "VMBUS:00"]


def add_numa(root: Path, node: int, cpulist: str) -> None:
    d = root / f"sys/devices/system/node/node{node}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpulist").write_text(cpulist + "\n")


def add_pci(root: Path, chain: list[str], bdf: str, vendor: str, cls: str, numa: int) -> Path:
    dev = root.joinpath("sys/devices", *chain, bdf)
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    (dev / "class").write_text(cls + "\n")
    (dev / "numa_node").write_text(f"{numa}\n")
    link = root / "sys/bus/pci/devices" / bdf
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(dev)
    return dev


def add_ib(
    root: Path,
    name: str,
    dev: Path | None,
    link_layer: str = "InfiniBand",
    rate: str = "400 Gb/sec (4X NDR)",
    state: str = "4: ACTIVE",
    parent: str = "",
) -> None:
    ib = root / "sys/class/infiniband" / name
    port = ib / "ports/1"
    port.mkdir(parents=True)
    (port / "state").write_text(state + "\n")
    (port / "link_layer").write_text(link_layer + "\n")
    (port / "rate").write_text(rate + "\n")
    (ib / "board_id").write_text("MT_0000000838\n")
    (ib / "fw_ver").write_text("28.39.1002\n")
    if parent:
        (ib / "parent").write_text(parent + "\n")
    if dev is not None:
        (ib / "device").symlink_to(dev)
        (dev / "net" / f"ib{name[-1]}").mkdir(parents=True)


def build_dgx(root: Path) -> None:
    """2 sockets x 2 top switches x 2 (GPU, NIC) pairs, each pair behind its own nested switch."""
    add_numa(root, 0, "0-55")
    add_numa(root, 1, "56-111")
    ni = 0
    for sock in (0, 1):
        for sw in (0, 1):
            bus = 0x10 + sock * 0x40 + sw * 0x20
            rc, rp, usp = f"pci0000:{bus:02x}", f"0000:{bus:02x}:01.0", f"0000:{bus + 1:02x}:00.0"
            for pair in (0, 1):
                dsp = f"0000:{bus + 2:02x}:{pair:02x}.0"
                usp2 = f"0000:{bus + 3 + pair * 4:02x}:00.0"
                b = bus + 4 + pair * 4
                add_pci(
                    root,
                    [rc, rp, usp, dsp, usp2, f"0000:{b:02x}:00.0"],
                    f"0000:{b + 1:02x}:00.0",
                    NVIDIA,
                    GPU_CLASS,
                    sock,
                )
                nic = add_pci(
                    root,
                    [rc, rp, usp, dsp, usp2, f"0000:{b:02x}:01.0"],
                    f"0000:{b + 2:02x}:00.0",
                    MELLANOX,
                    IB_CLASS,
                    sock,
                )
                add_ib(root, f"mlx5_{ni}", nic)
                ni += 1


def build_trap(root: Path) -> None:
    """8 GPUs, 4 NICs, 2 sockets; NIC names run opposite to socket order, so index pairing lands on SYS."""
    add_numa(root, 0, "0-31")
    add_numa(root, 1, "32-63")
    nic_names = {(0, 0): "mlx5_2", (0, 1): "mlx5_3", (1, 0): "mlx5_0", (1, 1): "mlx5_1"}
    for sock in (0, 1):
        for sw in (0, 1):
            bus = 0x10 + sock * 0x80 + sw * 0x20
            chain = [f"pci0000:{bus:02x}", f"0000:{bus:02x}:01.0", f"0000:{bus + 1:02x}:00.0"]
            for slot in (0, 1):
                add_pci(
                    root,
                    chain + [f"0000:{bus + 2:02x}:{slot:02x}.0"],
                    f"0000:{bus + 3 + slot:02x}:00.0",
                    NVIDIA,
                    GPU_CLASS,
                    sock,
                )
            nic = add_pci(
                root, chain + [f"0000:{bus + 2:02x}:02.0"], f"0000:{bus + 5:02x}:00.0", MELLANOX, IB_CLASS, sock
            )
            add_ib(root, nic_names[sock, sw], nic)


def build_4x8(root: Path) -> None:
    """One socket, 4 GPUs, 8 NICs: each switch holds one GPU and two NICs."""
    add_numa(root, 0, "0-63")
    ni = 0
    for sw in range(4):
        bus = 0x10 + sw * 0x10
        chain = [f"pci0000:{bus:02x}", f"0000:{bus:02x}:01.0", f"0000:{bus + 1:02x}:00.0"]
        add_pci(root, chain + [f"0000:{bus + 2:02x}:00.0"], f"0000:{bus + 3:02x}:00.0", NVIDIA, GPU_CLASS, 0)
        for slot in (1, 2):
            nic = add_pci(
                root,
                chain + [f"0000:{bus + 2:02x}:{slot:02x}.0"],
                f"0000:{bus + 3 + slot:02x}:00.0",
                MELLANOX,
                IB_CLASS,
                0,
            )
            add_ib(root, f"mlx5_{ni}", nic)
            ni += 1


AZURE_GPUS = ["0001", "0002", "0003", "0008", "0009", "000a", "000b", "000c"]


def build_azure_vm(root: Path, with_topo_file: bool = True) -> None:
    """ND H100 v5 guest: every device on its own VMBus PCI domain, plus the accelerated-networking NIC."""
    add_numa(root, 0, "0-47")
    add_numa(root, 1, "48-95")
    for i, dom in enumerate(AZURE_GPUS):
        guid = f"0000000{i}-0000-0000-0000-000000000000"
        add_pci(root, VMBUS_PREFIX + [guid, f"pci{dom}:00"], f"{dom}:00:00.0", NVIDIA, GPU_CLASS, i // 4)
        ndom = f"01{i + 1:02x}"
        nic = add_pci(root, VMBUS_PREFIX + [guid + "n", f"pci{ndom}:00"], f"{ndom}:00:00.0", MELLANOX, IB_CLASS, i // 4)
        add_ib(root, f"mlx5_ib{i}", nic)
    an = add_pci(root, VMBUS_PREFIX + ["anguid", "pcic0f4:00"], "c0f4:00:02.0", MELLANOX, "0x020000", 0)
    add_ib(root, "mlx5_an0", an, link_layer="Ethernet", rate="100 Gb/sec (2X HDR)")
    if with_topo_file:
        (root / "opt/microsoft").mkdir(parents=True)
        shutil.copy(FIXTURES / "azure-ndv5-topo.xml", root / "opt/microsoft/ndv5-topo.xml")


def add_cgroup_v2(root: Path, cpus: str, rel: str = "/", at: str | None = None) -> None:
    """This process sits in cgroup `rel`; `at` (default `rel`) is where cpuset.cpus.effective is visible."""
    proc = root / "proc/self"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "cgroup").write_text(f"0::{rel}\n")
    d = root / "sys/fs/cgroup" / (at if at is not None else rel).lstrip("/")
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpuset.cpus.effective").write_text(cpus + "\n")


def add_cgroup_v1(root: Path, cpus: str, rel: str = "/") -> None:
    proc = root / "proc/self"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "cgroup").write_text(f"12:memory:{rel}\n5:cpuset:{rel}\n0::/\n")
    d = root / "sys/fs/cgroup/cpuset" / rel.lstrip("/")
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpuset.effective_cpus").write_text(cpus + "\n")


def add_mems(root: Path, mems: str) -> None:
    proc = root / "proc/self"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "status").write_text(f"Name:\tpython\nCpus_allowed_list:\t0-95\nMems_allowed_list:\t{mems}\n")


def add_uverbs(root: Path, names: list[str], present: set[str]) -> None:
    """A verbs device per RDMA device; only those in `present` have a /dev/infiniband node (as in a container)."""
    (root / "dev/infiniband").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        d = root / f"sys/class/infiniband_verbs/uverbs{i}"
        d.mkdir(parents=True)
        (d / "ibdev").write_text(name + "\n")
        if name in present:
            (root / f"dev/infiniband/uverbs{i}").touch()
