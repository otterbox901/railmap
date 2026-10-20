"""Run nccl-tests with no placement, railmap's placement and a deliberately bad one, and compare bus bandwidth."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field

AVG_RE = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)")


@dataclass
class Run:
    variant: str
    test: str
    avg_busbw: float | None
    rows: list[tuple[int, float, float]] = field(default_factory=list)  # (bytes, time us, busbw GB/s) out-of-place
    exit_code: int = 0
    command: str = ""
    tail: str = ""


def parse(output: str) -> tuple[float | None, list[tuple[int, float, float]]]:
    m = AVG_RE.search(output)
    rows = []
    for line in output.splitlines():
        f = line.split()
        # size count type redop root | time algbw busbw #wrong | (in-place) time algbw busbw #wrong
        if len(f) >= 9 and f[0].isdigit() and f[1].isdigit():
            try:
                rows.append((int(f[0]), float(f[5]), float(f[7])))
            except ValueError:
                continue
    return (float(m.group(1)) if m else None), rows


def command(launcher: str, variant: str, binary: str, args: list[str], exec_args: list[str]) -> list[str]:
    """Each rank plans on its own node (`exec --variant`), so no plan file has to be copied between nodes."""
    cmd = shlex.split(launcher)
    if variant != "none":
        cmd += ["railmap", "exec", "--variant", variant, *exec_args, "--"]
    return cmd + [binary, *args]


def run(launcher: str, variant: str, binary: str, args: list[str], exec_args: list[str], log=print) -> Run:
    cmd = command(launcher, variant, binary, args, exec_args)
    log("$ " + shlex.join(cmd))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    avg, rows = parse(out)
    return Run(variant, binary.rsplit("/", 1)[-1], avg, rows, proc.returncode, shlex.join(cmd), out[-3000:])
