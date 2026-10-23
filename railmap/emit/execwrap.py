"""`railmap exec`: bind the calling rank per the plan, then exec the real command."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping

from railmap.emit.env import rank_env
from railmap.placement.assign import Plan
from railmap.topology import cpuset

# torchrun, Open MPI, Slurm, MPICH/Intel MPI, MVAPICH
LOCAL_RANK_VARS = (
    "LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",
)


def detect_local_rank(env: Mapping[str, str]) -> int:
    for var in LOCAL_RANK_VARS:
        if env.get(var, "").isdigit():
            return int(env[var])
    raise RuntimeError(f"cannot find the local rank: none of {', '.join(LOCAL_RANK_VARS)} is set; pass --local-rank")


def build(
    plan: Plan,
    cmd: list[str],
    local_rank: int,
    env: Mapping[str, str],
    isolate_gpu: bool,
    mems: set[int] | None = None,
) -> tuple[list[str], dict, list[int]]:
    """Return (argv, env, cpus) without side effects, so it can be tested. `mems`: NUMA nodes allowed, if known."""
    r = plan.rank(local_rank)
    new_env = {**env, **rank_env(plan, local_rank, isolate_gpu)}
    argv = list(cmd)
    if plan.numa_count > 1 and r.numa >= 0:
        if mems is not None and r.numa not in mems:
            print(
                f"railmap: NUMA node {r.numa} is not in this cgroup's memory nodes; memory is not bound",
                file=sys.stderr,
            )
        elif shutil.which("numactl"):
            argv = ["numactl", f"--membind={r.numa}", *argv]
        else:
            print("railmap: numactl not found; memory is not NUMA-bound", file=sys.stderr)
    return argv, new_env, r.cpus


def run(plan: Plan, cmd: list[str], local_rank: int | None = None, isolate_gpu: bool = False) -> None:
    rank = local_rank if local_rank is not None else detect_local_rank(os.environ)
    argv, env, cpus = build(plan, cmd, rank, os.environ, isolate_gpu, cpuset.allowed_mems())
    allowed = os.sched_getaffinity(0)
    usable = set(cpus) & allowed
    if usable:
        os.sched_setaffinity(0, usable)
        if len(usable) < len(cpus):
            print(
                f"railmap: only {len(usable)}/{len(cpus)} plan CPUs are allowed by the cgroup/launcher", file=sys.stderr
            )
    else:
        print("railmap: none of the plan CPUs are allowed here; leaving affinity unchanged", file=sys.stderr)
    os.execvpe(argv[0], argv, env)
