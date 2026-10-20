"""Compare a job's QPs per NIC with a NIC calibration profile and suggest settings that fit under the knee."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from railmap.placement.connections import Job, estimate, max_nodes_under
from railmap.topology.model import Nic

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


@dataclass
class Profile:
    name: str
    kind: str  # "prior" | "measured"
    knee_qps: int
    curve: list[tuple[int, float]]  # (QPs, message rate relative to peak)
    match: dict
    source: str = ""
    path: str = ""

    def matches(self, nic: Nic | None) -> bool:
        if not self.match:
            return True
        if nic is None:
            return False
        return all(getattr(nic, k, None) == v for k, v in self.match.items())

    def relative_rate(self, qps: int) -> float | None:
        pts = sorted(self.curve)
        if not pts or qps <= 0:
            return None
        if qps <= pts[0][0]:
            return pts[0][1] if self.kind == "measured" or qps == pts[0][0] else None
        if qps >= pts[-1][0]:
            return pts[-1][1]
        for (q0, r0), (q1, r1) in itertools.pairwise(pts):
            if q0 <= qps <= q1:
                t = (math.log(qps) - math.log(q0)) / (math.log(q1) - math.log(q0))
                return r0 + t * (r1 - r0)
        return None


def load_profile(path: Path) -> Profile:
    d = json.loads(path.read_text())
    return Profile(
        name=d["name"],
        kind=d["kind"],
        knee_qps=int(d["knee_qps"]),
        curve=[(int(q), float(r)) for q, r in d["curve"]],
        match=d.get("match", {}),
        source=d.get("source", ""),
        path=str(path),
    )


def select_profile(nic: Nic | None, extra: Path | None = None) -> Profile:
    """An explicit profile wins; otherwise the most specific matching shipped profile, measured before prior."""
    if extra:
        return load_profile(extra)
    profiles = [load_profile(p) for p in sorted(PROFILE_DIR.glob("*.json"))]
    candidates = [p for p in profiles if p.matches(nic)]
    candidates.sort(key=lambda p: (p.kind != "measured", -len(p.match)))
    return candidates[0]


@dataclass
class Recommendation:
    change: str
    setting: str
    qps_per_nic: int
    max_nodes: int | None


@dataclass
class Budget:
    job: Job
    qps_per_nic: int
    qps_per_gpu: int
    formula: str
    profile: str
    profile_kind: str
    knee_qps: int
    per_gpu_budget: int
    status: str  # "ok" | "near" | "over"
    relative_rate: float | None
    beyond_curve: bool  # QPs exceed the last profile point; the rate is then an upper bound
    max_nodes: int | None
    recommendations: list[Recommendation]


def _candidates(job: Job) -> list[tuple[str, str, Job]]:
    out = []
    if job.pattern == "alltoall" and not job.rail_only:
        out.append(
            (
                "hierarchical all-to-all: RDMA only to the same-rail GPU on each node, forward over NVLink",
                "e.g. DeepEP internode kernels; cuts peers per GPU by gpus_per_node",
                replace(job, rail_only=True),
            )
        )
    if job.stack == "nccl":
        if job.qps_per_conn > 1:
            out.append(("one QP per connection", "NCCL_IB_QPS_PER_CONNECTION=1", replace(job, qps_per_conn=1)))
        if job.pattern == "alltoall" and job.channels_per_peer > 1:
            out.append(
                (
                    "one network channel per peer",
                    "NCCL_NCHANNELS_PER_NET_PEER=1 (undocumented NCCL parameter; check your version)",
                    replace(job, channels_per_peer=1),
                )
            )
        if job.pattern == "allreduce" and job.channels > 4:
            out.append(("fewer channels", "NCCL_MAX_CTAS=4 (NCCL_MAX_NCHANNELS before 2.17)", replace(job, channels=4)))
    else:
        if not job.dc:
            out.append(
                (
                    "DC transport instead of RC (QPs independent of peer count)",
                    "NVSHMEM IBGDA DC mode, NVSHMEM_IBGDA_NUM_DCI",
                    replace(job, dc=True),
                )
            )
            if job.rc_per_peer > 1:
                out.append(
                    (
                        "one RC QP per peer",
                        "NVSHMEM_IBGDA_NUM_RC_PER_PE=1 (DeepEP: num_qps_per_rank)",
                        replace(job, rc_per_peer=1),
                    )
                )
    return out


def evaluate(job: Job, profile: Profile) -> Budget:
    est = estimate(job)
    knee = profile.knee_qps
    status = "ok" if est.qps_per_nic < knee / 2 else "near" if est.qps_per_nic < knee else "over"
    recs = []
    if status != "ok":
        for change, setting, alt in _candidates(job):
            alt_qps = estimate(alt).qps_per_nic
            if alt_qps < est.qps_per_nic:
                recs.append(Recommendation(change, setting, alt_qps, max_nodes_under(alt, knee)))
        recs.sort(key=lambda r: r.qps_per_nic)
    gpus_per_nic = max(1, math.ceil(job.gpus_per_node / job.nics_per_node))
    return Budget(
        job=job,
        qps_per_nic=est.qps_per_nic,
        qps_per_gpu=est.qps_per_gpu,
        formula=est.formula,
        profile=profile.name,
        profile_kind=profile.kind,
        knee_qps=knee,
        per_gpu_budget=knee // gpus_per_nic,
        status=status,
        relative_rate=profile.relative_rate(est.qps_per_nic),
        beyond_curve=bool(profile.curve) and est.qps_per_nic > max(q for q, _ in profile.curve),
        max_nodes=max_nodes_under(job, knee),
        recommendations=recs,
    )
