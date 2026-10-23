"""CPUs and NUMA memory nodes this process may use: the cgroup cpuset first, then the scheduler affinity.

The cgroup is shared by every rank in a pod, container or Slurm step, while affinity differs per rank once a launcher
pins processes, so plans built from the cgroup agree across ranks. Every function takes `root` so tests can fake it.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from railmap.topology.model import parse_cpulist

V2_FILE = "cpuset.cpus.effective"
V1_FILE = "cpuset.effective_cpus"


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _first_up(base: Path, rel: str, name: str) -> str:
    """`name` in the cgroup dir `rel` under `base`, or its nearest ancestor that has it.

    Without a cgroup namespace (cgroup v1 Docker) /proc/self/cgroup names a host path that does not exist inside the
    container, whose own cgroup is mounted at `base`; walking up lands there.
    """
    p = PurePosixPath("/") / rel
    for d in [p, *p.parents]:
        text = _read(base / str(d).lstrip("/") / name)
        if text:
            return text
    return ""


def cgroup_cpus(root: Path = Path("/")) -> list[int] | None:
    """Effective cpuset of this process's cgroup (v2 or v1), or None if there is none to read."""
    for line in _read(root / "proc/self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        if hierarchy == "0" and controllers == "":
            text = _first_up(root / "sys/fs/cgroup", rel, V2_FILE)
        elif "cpuset" in controllers.split(","):
            text = _first_up(root / "sys/fs/cgroup/cpuset", rel, V1_FILE)
        else:
            continue
        if text:
            return parse_cpulist(text)
    return None


def allowed_cpus(root: Path = Path("/")) -> tuple[set[int], str] | None:
    """(CPUs, where the limit comes from), or None when nothing can be read."""
    cpus = cgroup_cpus(root)
    if cpus:
        return set(cpus), "cgroup"
    if root == Path("/"):
        return set(os.sched_getaffinity(0)), "CPU affinity"
    return None


def allowed_mems(root: Path = Path("/")) -> set[int] | None:
    """NUMA nodes this process may allocate from (Mems_allowed_list), or None if unknown."""
    for line in _read(root / "proc/self/status").splitlines():
        if line.startswith("Mems_allowed_list:"):
            return set(parse_cpulist(line.split(":", 1)[1]))
    return None
