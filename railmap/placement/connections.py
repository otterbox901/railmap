"""Estimate RDMA queue pairs (QPs) per NIC for a collective job.

These are estimates of NCCL/NVSHMEM internals; `railmap observe` measures the real count with `rdma res show qp`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Job:
    nodes: int
    gpus_per_node: int
    nics_per_node: int
    pattern: str = "alltoall"  # "alltoall" | "allreduce"
    stack: str = "nccl"  # "nccl" | "nvshmem"
    # NCCL: network channels per peer (send/recv) and NCCL_IB_QPS_PER_CONNECTION
    channels_per_peer: int = 2
    qps_per_conn: int = 1
    # NCCL all-reduce channels (rings/trees)
    channels: int = 16
    # NVSHMEM IBGDA: RC QPs per PE per peer (DeepEP sets NVSHMEM_IBGDA_NUM_RC_PER_PE), or DC initiators per PE
    rc_per_peer: int = 2
    dc: bool = False
    dcis_per_pe: int = 4
    # Hierarchical all-to-all: talk over RDMA only to the same-rail GPU on each remote node, forward over NVLink
    rail_only: bool = False

    def __post_init__(self) -> None:
        if self.pattern not in ("alltoall", "allreduce"):
            raise ValueError(f"unknown pattern {self.pattern!r}")
        if self.stack not in ("nccl", "nvshmem"):
            raise ValueError(f"unknown stack {self.stack!r}")
        if min(self.nodes, self.gpus_per_node, self.nics_per_node) < 1:
            raise ValueError("nodes, gpus_per_node and nics_per_node must be >= 1")


@dataclass(frozen=True)
class Estimate:
    qps_per_gpu: int
    qps_per_nic: int
    remote_peers_per_gpu: int
    formula: str


def remote_peers(job: Job) -> int:
    if job.pattern == "allreduce":
        return 0  # handled per channel, not per peer
    if job.rail_only:
        return job.nodes - 1
    return job.gpus_per_node * (job.nodes - 1)


def estimate(job: Job) -> Estimate:
    gpus_per_nic = job.gpus_per_node / job.nics_per_node
    peers = remote_peers(job)
    if job.nodes == 1:
        return Estimate(0, 0, 0, "single node: no inter-node QPs")
    if job.pattern == "allreduce":
        # Each ring/tree channel enters and leaves the node once: one send + one recv connection per channel.
        per_node = 2 * job.channels * job.qps_per_conn * 2  # rings + trees
        per_nic = math.ceil(per_node / job.nics_per_node)
        return Estimate(
            math.ceil(per_node / job.gpus_per_node),
            per_nic,
            2,
            f"2 dirs x {job.channels} ch x {job.qps_per_conn} qp x 2 algos / {job.nics_per_node} NICs",
        )
    if job.stack == "nvshmem":
        if job.dc:
            per_gpu = job.dcis_per_pe + 1  # initiators plus one DC target
            formula = f"{job.dcis_per_pe} DCIs + 1 DCT per PE (independent of peers)"
        else:
            per_gpu = peers * job.rc_per_peer
            formula = f"{peers} peers x {job.rc_per_peer} RC/peer"
    else:
        # NCCL builds a separate connection per direction, each with qps_per_conn QPs, per network channel.
        per_gpu = peers * job.channels_per_peer * job.qps_per_conn * 2
        formula = f"{peers} peers x {job.channels_per_peer} ch x {job.qps_per_conn} qp x 2 dirs"
    per_nic = math.ceil(per_gpu * gpus_per_nic)
    if gpus_per_nic != 1:
        formula += f" x {gpus_per_nic:g} GPUs/NIC"
    return Estimate(per_gpu, per_nic, peers, formula)


def max_nodes_under(job: Job, limit: int, cap: int = 100_000) -> int | None:
    """Largest node count whose per-NIC QP estimate stays below `limit`; None if it never crosses within `cap`."""
    if estimate(replace(job, nodes=cap)).qps_per_nic < limit:
        return None
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(replace(job, nodes=mid)).qps_per_nic < limit:
            lo = mid
        else:
            hi = mid - 1
    return lo
