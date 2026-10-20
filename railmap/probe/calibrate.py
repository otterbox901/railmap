"""Measure small-message rate against QP count and fit the knee into a NIC profile."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from railmap.probe import perftest
from railmap.topology.model import Nic

DEFAULT_QPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
KNEE_FRACTION = 0.9  # knee: first QP count after the peak where message rate falls below 90% of peak


@dataclass
class Point:
    qps: int
    msg_rate_mpps: float
    bw_avg_gbps: float


def sweep(
    server: perftest.Endpoint,
    client: perftest.Endpoint,
    connect_to: str,
    qps_list: list[int],
    tool: str = "ib_write_bw",
    size: int = 64,
    duration: int = 5,
    tx_depth: int = 16,
    base_port: int = 18515,
    log=print,
) -> list[Point]:
    points = []
    for i, q in enumerate(qps_list):
        common = ["-q", str(q), "-s", str(size), "-D", str(duration), "-t", str(tx_depth)]
        r = perftest.run_pair(tool, server, client, connect_to, common, base_port + i)
        points.append(Point(q, r.msg_rate_mpps, r.bw_avg_gbps))
        log(f"qps={q:>5}  {r.msg_rate_mpps:8.2f} Mpps  {r.bw_avg_gbps:7.2f} Gb/s")
    return points


def fit(points: list[Point]) -> tuple[int | None, list[tuple[int, float]]]:
    """Return (knee QPs or None if the rate never drops, relative-rate curve)."""
    pts = sorted(points, key=lambda p: p.qps)
    peak_i = max(range(len(pts)), key=lambda i: pts[i].msg_rate_mpps)
    peak = pts[peak_i].msg_rate_mpps
    curve = [(p.qps, round(p.msg_rate_mpps / peak, 4)) for p in pts]
    for i in range(peak_i + 1, len(pts)):
        r0, r1 = curve[i - 1][1], curve[i][1]
        if r1 < KNEE_FRACTION:
            q0, q1 = pts[i - 1].qps, pts[i].qps
            t = (r0 - KNEE_FRACTION) / (r0 - r1) if r0 != r1 else 1.0
            return round(math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))), curve
    return None, curve


def to_profile(points: list[Point], nic: Nic | None, hca: str, tool: str, size: int, name: str = "") -> dict:
    knee, curve = fit(points)
    match = {}
    if nic is not None:
        match = {k: v for k, v in (("board_id", nic.board_id), ("link_layer", nic.link_layer)) if v}
    stamp = time.strftime("%Y-%m-%d")
    return {
        "name": name or f"measured-{(nic.board_id if nic and nic.board_id else hca)}-{stamp}",
        "kind": "measured",
        "match": match,
        # Without a drop inside the sweep, the largest tested count is the only safe bound.
        "knee_qps": knee if knee is not None else max(p.qps for p in points),
        "knee_found": knee is not None,
        "curve": curve,
        "source": f"railmap calibrate {stamp}: {tool} -s {size}, {hca}"
        + (f" fw {nic.fw_ver}" if nic and nic.fw_ver else ""),
        "raw": [[p.qps, p.msg_rate_mpps, p.bw_avg_gbps] for p in points],
    }
