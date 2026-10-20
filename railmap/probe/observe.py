"""Count live RDMA QPs per NIC with `rdma -j resource show qp` while a job runs."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def sample_qps() -> list[dict]:
    out = subprocess.run(["rdma", "-j", "resource", "show", "qp"], capture_output=True, text=True, check=True).stdout
    return json.loads(out or "[]")


def qp_counts(entries: list[dict], pids: set[int] | None = None) -> Counter[str]:
    """QPs per "nic:port"; with `pids`, only QPs owned by those processes (kernel QPs have no pid)."""
    c: Counter[str] = Counter()
    for e in entries:
        if pids is not None and e.get("pid") not in pids:
            continue
        c[f"{e.get('ifname', '?')}:{e.get('port', 1)}"] += 1
    return c


def descendants(pid: int, proc: Path = Path("/proc")) -> set[int]:
    children: dict[int, list[int]] = {}
    for stat in proc.glob("[0-9]*/stat"):
        try:
            # comm may contain spaces and parens; ppid is the 2nd field after the last ')'
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    out, stack = {pid}, [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


@dataclass
class Observation:
    peak_per_nic: dict[str, int] = field(default_factory=dict)
    peak_total: int = 0
    timeline: list[tuple[float, int]] = field(default_factory=list)  # (seconds since start, total QPs)
    exit_code: int | None = None
    mode: str = "pid"


def _update(obs: Observation, counts: Counter[str], t: float) -> None:
    for nic, n in counts.items():
        obs.peak_per_nic[nic] = max(obs.peak_per_nic.get(nic, 0), n)
    total = sum(counts.values())
    obs.peak_total = max(obs.peak_total, total)
    obs.timeline.append((round(t, 2), total))


def observe_command(cmd: list[str], interval: float = 0.5) -> Observation:
    """Run `cmd` and track QPs owned by it and its descendants."""
    obs = Observation(mode="pid")
    start = time.monotonic()
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        _update(obs, qp_counts(sample_qps(), descendants(proc.pid)), time.monotonic() - start)
        time.sleep(interval)
    obs.exit_code = proc.returncode
    return obs


def observe_host(duration: float, interval: float = 0.5) -> Observation:
    """Track all QPs on this host for `duration` seconds, minus those present at the start (for remote nodes)."""
    obs = Observation(mode="host")
    baseline = qp_counts(sample_qps())
    start = time.monotonic()
    while (t := time.monotonic() - start) < duration:
        counts = qp_counts(sample_qps())
        counts.subtract(baseline)
        _update(obs, +counts, t)
        time.sleep(interval)
    return obs
