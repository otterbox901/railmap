"""`railmap doctor`: check that this host, container or job step has what railmap and RDMA jobs need.

Every check takes `root`, `env` and friends so tests can fake the host. Checks never raise; a broken host is a "fail".
"""

from __future__ import annotations

import os
import platform
import resource
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from railmap.emit.execwrap import LOCAL_RANK_VARS
from railmap.probe import perftest
from railmap.topology import cpuset, sysfs
from railmap.topology.discover import discover, placement_nics
from railmap.topology.model import Topology, parse_cpulist

OK, WARN, FAIL = "ok", "warn", "fail"
MEMLOCK_MIN = 64 << 30  # bytes; below this, large RDMA registrations (NCCL buffers, GDR) start failing


@dataclass
class Check:
    name: str
    status: str  # ok, warn or fail
    detail: str
    fix: str = ""


def version() -> str:
    try:
        return metadata.version("railmap")
    except metadata.PackageNotFoundError:
        return "unknown"


def check_python() -> Check:
    return Check("railmap", OK, f"railmap {version()}, Python {platform.python_version()} ({sys.executable})")


def check_container(root: Path, env: Mapping[str, str]) -> Check:
    if env.get("KUBERNETES_SERVICE_HOST"):
        kind = "Kubernetes pod"
    elif (root / "run/.containerenv").exists():
        kind = "Podman container"
    elif (root / ".dockerenv").exists():
        kind = "Docker container"
    elif env.get("container"):
        kind = f"{env['container']} container"
    elif env.get("ENROOT_ROOTFS") or env.get("PYXIS_RUNTIME_PATH"):
        kind = "Enroot container"
    else:
        kind = "no container detected"
    if env.get("SLURM_JOB_ID"):
        kind += f", Slurm job {env['SLURM_JOB_ID']}"
    return Check("container", OK, kind)


def check_gpus(topo: Topology, nvml_names: int) -> Check:
    if not topo.gpus:
        return Check(
            "GPUs",
            FAIL,
            "no NVIDIA GPUs in /sys/bus/pci",
            "load the NVIDIA driver; railmap plans need at least one GPU",
        )
    if nvml_names == 0:
        return Check(
            "GPUs",
            WARN,
            f"{len(topo.gpus)} in sysfs, but NVML is unavailable: GPU names are unknown, and in a container railmap "
            "can't tell which GPUs are yours",
            "Docker: --gpus all; Kubernetes: request nvidia.com/gpu; or install the NVIDIA driver",
        )
    return Check("GPUs", OK, f"{len(topo.gpus)} usable ({topo.gpus[0].name or 'unknown model'})")


def check_nics(topo: Topology) -> Check:
    if not topo.nics:
        return Check(
            "RDMA NICs",
            WARN,
            "none in /sys/class/infiniband: plans cover CPU and NUMA binding only",
            "load mlx5_ib (or your NIC's RDMA driver); on a laptop, scripts/softroce-up.sh gives a test device",
        )
    usable = placement_nics(topo)
    kinds = sorted({n.link_layer or "unknown link" for n in usable})
    rates = sorted({f"{n.rate_gbps:g} Gb/s" for n in usable if n.rate_gbps})
    detail = f"{len(usable)} for placement of {len(topo.nics)} ports ({', '.join(kinds + rates)})"
    if not usable:
        return Check(
            "RDMA NICs", WARN, f"{len(topo.nics)} ports, none active", "check `ibstat` / cabling / subnet manager"
        )
    return Check("RDMA NICs", OK, detail)


def check_verbs(root: Path, topo: Topology) -> Check:
    verbs = sysfs.uverbs_devices(root)
    if not topo.nics:
        return Check("verbs devices", OK, "no RDMA NICs, nothing to open")
    if not verbs:
        return Check(
            "verbs devices", WARN, "no /sys/class/infiniband_verbs: ib_uverbs is not loaded", "modprobe ib_uverbs"
        )
    opened = sorted(name for name, ok in verbs.items() if ok)
    if not opened:
        return Check(
            "verbs devices",
            FAIL,
            f"none of {len(verbs)} /dev/infiniband/uverbs* nodes exist here: RDMA can't open any NIC",
            "Docker: --device /dev/infiniband; Kubernetes: request an RDMA resource from the NVIDIA Network Operator",
        )
    denied = [n for n in opened if not _rw(root, n)]
    if denied:
        return Check(
            "verbs devices", FAIL, f"no read/write access to {', '.join(denied)}", "check /dev/infiniband permissions"
        )
    return Check("verbs devices", OK, f"{len(opened)} of {len(verbs)} openable: {', '.join(opened)}")


def _rw(root: Path, name: str) -> bool:
    for d in (root / "sys/class/infiniband_verbs").glob("uverbs*"):
        if _read(d / "ibdev") == name:
            return os.access(root / "dev/infiniband" / d.name, os.R_OK | os.W_OK)
    return False


def check_topology(topo: Topology) -> Check:
    if topo.warnings():
        return Check(
            "topology",
            FAIL,
            "virtualized PCI tree and no matching topology file: GPU-NIC path classes would be wrong",
            "pass --topo-file (or set NCCL_TOPO_FILE); in a container, mount the host's /opt/microsoft read-only",
        )
    detail = topo.source + (" (virtualized PCI tree)" if topo.flattened else "")
    return Check("topology", OK, detail)


def check_cpus(root: Path, topo: Topology) -> Check:
    online = parse_cpulist(_read(root / "sys/devices/system/cpu/online")) or list(range(os.cpu_count() or 1))
    allowed = cpuset.allowed_cpus(root)
    if allowed is None:
        return Check("CPUs", OK, f"{len(online)} online; no cpuset limit found")
    cpus, why = allowed
    gpu_nodes = {g.numa for g in topo.gpus if g.numa >= 0}
    empty = sorted(n.id for n in topo.numa_nodes if n.id in gpu_nodes and not set(n.cpus) & cpus)
    detail = f"{len(cpus & set(online))} of {len(online)} allowed by the {why}"
    if empty:
        return Check(
            "CPUs",
            WARN,
            f"{detail}; none on NUMA node(s) {', '.join(map(str, empty))}, where GPUs sit",
            "Kubernetes: use the static CPU manager with the single-numa-node or restricted topology policy",
        )
    return Check("CPUs", OK, detail)


def check_mems(root: Path, topo: Topology) -> Check:
    mems = cpuset.allowed_mems(root)
    nodes = {n.id for n in topo.numa_nodes}
    if mems is None or nodes <= mems:
        return Check("NUMA memory", OK, f"{len(nodes)} node(s), all allowed")
    missing = sorted(nodes - mems)
    return Check(
        "NUMA memory",
        WARN,
        f"cgroup memory nodes exclude {', '.join(map(str, missing))}: exec won't bind memory for ranks there",
        "allow all memory nodes in the container's cpuset.mems",
    )


def check_memlock(limit: int | None = None) -> Check:
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] if limit is None else limit
    if soft == resource.RLIM_INFINITY:
        return Check("locked memory", OK, "unlimited")
    if soft >= MEMLOCK_MIN:
        return Check("locked memory", OK, f"{soft >> 30} GiB")
    shown = f"{soft >> 20} MiB" if soft >= 1 << 20 else f"{soft >> 10} KiB"
    return Check(
        "locked memory",
        WARN,
        f"ulimit -l is {shown}: RDMA memory registration (NCCL, GPUDirect) may fail",
        "Docker: --ulimit memlock=-1; Kubernetes: raise the container runtime's default; Slurm: "
        "PropagateResourceLimitsExcept=MEMLOCK",
    )


def check_tools(
    which: Callable[[str], str | None], numa_nodes: int, cuda: Callable[[str], bool] = perftest.supports_cuda
) -> list[Check]:
    out = []
    if which("numactl"):
        out.append(Check("numactl", OK, which("numactl") or ""))
    else:
        status = WARN if numa_nodes > 1 else OK
        out.append(
            Check("numactl", status, "missing: exec can't bind memory to the GPU's NUMA node", "install numactl")
        )
    rdma = which("rdma")
    out.append(
        Check("rdma (iproute2)", OK, rdma)
        if rdma
        else Check("rdma (iproute2)", WARN, "missing: `railmap observe` needs it", "install iproute2")
    )
    bw = which("ib_write_bw")
    if not bw:
        out.append(Check("perftest", WARN, "missing: calibrate and gdr-matrix need it", "install perftest"))
    elif not cuda(bw):
        out.append(
            Check("perftest", WARN, f"{bw} has no CUDA support: gdr-matrix needs --use_cuda", "rebuild perftest")
        )
    else:
        out.append(Check("perftest", OK, f"{bw} (CUDA)"))
    smi = which("nvidia-smi")
    out.append(Check("nvidia-smi", OK, smi or "not found (not needed by railmap)"))
    return out


def check_launcher(env: Mapping[str, str]) -> Check:
    for var in LOCAL_RANK_VARS:
        if env.get(var, "").isdigit():
            return Check("launcher", OK, f"{var}={env[var]}")
    return Check("launcher", OK, "no local-rank variable set (fine outside a launcher; exec then needs --local-rank)")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def run(
    topo_file: Path | None = None,
    root: Path = Path("/"),
    env: Mapping[str, str] = os.environ,
    which: Callable[[str], str | None] = shutil.which,
    memlock: int | None = None,
) -> list[Check]:
    checks = [check_python(), check_container(root, env)]
    try:
        topo = discover(topo_file=topo_file, root=root, env=env)
    except Exception as e:  # noqa: BLE001 - doctor reports, it doesn't crash
        checks.append(Check("topology", FAIL, f"discovery failed: {e}"))
        return checks + [check_memlock(memlock), *check_tools(which, 1), check_launcher(env)]
    checks += [
        check_gpus(topo, sum(1 for g in topo.gpus if g.uuid)),
        check_nics(topo),
        check_verbs(root, topo),
        check_topology(topo),
        check_cpus(root, topo),
        check_mems(root, topo),
        check_memlock(memlock),
        *check_tools(which, len(topo.numa_nodes)),
        check_launcher(env),
    ]
    return checks
