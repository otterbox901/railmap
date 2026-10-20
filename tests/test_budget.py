from dataclasses import replace

from railmap.placement.budget import Profile, evaluate, select_profile
from railmap.placement.connections import Job, estimate, max_nodes_under

JOB = Job(nodes=2, gpus_per_node=8, nics_per_node=8)


def test_nccl_alltoall_estimate():
    assert estimate(JOB).qps_per_nic == 8 * 2 * 1 * 2
    assert estimate(replace(JOB, nodes=128)).qps_per_nic == 127 * 8 * 2 * 2
    assert estimate(replace(JOB, nodes=1)).qps_per_nic == 0


def test_shared_nics_and_rail_only():
    assert estimate(replace(JOB, nics_per_node=4)).qps_per_nic == 2 * estimate(JOB).qps_per_nic
    assert estimate(replace(JOB, nodes=9, rail_only=True)).qps_per_nic == 8 * 2 * 2


def test_nvshmem_rc_and_dc():
    j = replace(JOB, stack="nvshmem", nodes=33, rc_per_peer=4)
    assert estimate(j).qps_per_nic == 256 * 4
    assert estimate(replace(j, dc=True)).qps_per_nic == estimate(replace(j, dc=True, nodes=500)).qps_per_nic


def test_max_nodes_under():
    assert max_nodes_under(JOB, 3000) == 94  # 32 * (n - 1) < 3000
    assert max_nodes_under(replace(JOB, stack="nvshmem", dc=True), 3000) is None


def test_prior_profile():
    prior = select_profile(None)
    assert prior.kind == "prior" and prior.knee_qps == 3000
    assert prior.relative_rate(100) is None
    assert prior.relative_rate(3000) == 0.41
    b = evaluate(replace(JOB, nodes=128), prior)
    assert b.status == "over" and b.relative_rate == 0.41 and b.beyond_curve
    assert b.recommendations[0].change.startswith("hierarchical")
    assert all(r.qps_per_nic < b.qps_per_nic for r in b.recommendations)


def test_ok_job_has_no_recommendations():
    b = evaluate(JOB, select_profile(None))
    assert b.status == "ok" and b.recommendations == []


def test_measured_profile_interpolates_in_log_space():
    p = Profile(name="m", kind="measured", knee_qps=1000, curve=[(1, 0.5), (100, 1.0), (10000, 0.4)], match={})
    assert p.relative_rate(1) == 0.5
    assert abs(p.relative_rate(1000) - 0.7) < 1e-9
    assert p.relative_rate(20000) == 0.4
