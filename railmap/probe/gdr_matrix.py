"""GPUDirect RDMA bandwidth for every GPU x NIC pair, to measure what each path class costs on this machine."""

from __future__ import annotations

from dataclasses import dataclass

from railmap.probe import perftest
from railmap.topology.model import Gpu, Nic, Topology


@dataclass
class Cell:
    gpu: int
    nic: str
    path: str
    bw_avg_gbps: float | None
    error: str = ""


def measure(
    topo: Topology,
    gpus: list[Gpu],
    nics: list[Nic],
    remote: str = "",
    connect_to: str = "127.0.0.1",
    tool: str = "ib_write_bw",
    size: int = 1 << 20,
    duration: int = 3,
    gid_index: int | None = None,
    base_port: int = 19515,
    log=print,
) -> list[Cell]:
    """Client sends from GPU memory through each NIC; the server (local or `remote`) receives into host memory."""
    cells = []
    k = 0
    for g in gpus:
        for n in nics:
            client = perftest.Endpoint(n.name, n.port, gid_index=gid_index, use_cuda=g.index)
            server = perftest.Endpoint(n.name, n.port, host=remote, gid_index=gid_index)
            path = topo.path(g, n).name
            try:
                r = perftest.run_pair(
                    tool, server, client, connect_to, ["-s", str(size), "-D", str(duration)], base_port + k
                )
                cells.append(Cell(g.index, n.label, path, r.bw_avg_gbps))
                log(f"GPU{g.index} -> {n.label} ({path}): {r.bw_avg_gbps:.1f} Gb/s")
            except (RuntimeError, ValueError) as e:
                cells.append(Cell(g.index, n.label, path, None, str(e)[:500]))
                log(f"GPU{g.index} -> {n.label} ({path}): failed")
            k += 1
    return cells


def by_path_class(cells: list[Cell]) -> dict[str, tuple[float, float, int]]:
    """Mean, min and count of bandwidth per path class."""
    out: dict[str, list[float]] = {}
    for c in cells:
        if c.bw_avg_gbps is not None:
            out.setdefault(c.path, []).append(c.bw_avg_gbps)
    return {p: (sum(v) / len(v), min(v), len(v)) for p, v in out.items()}
