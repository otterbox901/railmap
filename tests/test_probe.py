from collections import Counter
from pathlib import Path

from railmap.probe import calibrate, observe, perftest, validate
from railmap.topology.model import Nic

PERFTEST_OUT = """\
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF\t\tDevice         : mlx5_ib0
 Number of qps   : 1024\t\tTransport type : IB
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 64         51234000         0.00               26.23 \t\t   51.234000
---------------------------------------------------------------------------------------
"""

NCCL_TESTS_OUT = """\
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    35.11    0.00    0.00      0    34.84    0.00    0.00      0
     1048576        262144     float     sum      -1    45.12   23.24   43.57      0    44.80   23.41   43.89      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.785
"""


def test_parse_perftest():
    r = perftest.parse(PERFTEST_OUT)
    assert (r.size, r.bw_avg_gbps, r.msg_rate_mpps) == (64, 26.23, 51.234)


def test_endpoint_args_and_ssh_wrap():
    e = perftest.Endpoint("mlx5_0", 1, host="node2", gid_index=3, use_cuda=2)
    assert e.args() == ["-d", "mlx5_0", "-i", "1", "-x", "3", "--use_cuda=2"]
    assert e.wrap(["ib_write_bw", "-q", "8"]) == ["ssh", "-o", "BatchMode=yes", "node2", "ib_write_bw -q 8"]
    assert perftest.Endpoint("rxe0").wrap(["x"]) == ["x"]


def test_fit_finds_knee_after_peak():
    rates = {1: 20.0, 8: 90.0, 64: 100.0, 512: 98.0, 1024: 95.0, 2048: 70.0, 4096: 41.0}
    knee, curve = calibrate.fit([calibrate.Point(q, r, 0.0) for q, r in rates.items()])
    assert 1024 < knee < 2048
    assert dict(curve)[64] == 1.0 and dict(curve)[4096] == 0.41


def test_fit_without_drop():
    knee, _ = calibrate.fit([calibrate.Point(q, 50.0, 0.0) for q in (1, 2, 4)])
    assert knee is None


def test_profile_keys_on_nic_model():
    nic = Nic("mlx5_ib0", "0101:00:00.0", 0, [], board_id="MT_0000000838", link_layer="InfiniBand", fw_ver="28.1")
    pts = [calibrate.Point(q, r, 0.0) for q, r in ((1, 10.0), (100, 20.0), (5000, 5.0))]
    p = calibrate.to_profile(pts, nic, "mlx5_ib0", "ib_write_bw", 64)
    assert p["match"] == {"board_id": "MT_0000000838", "link_layer": "InfiniBand"}
    assert p["kind"] == "measured" and p["knee_found"] and "fw 28.1" in p["source"]


def test_qp_counts_by_pid():
    entries = [
        {"ifname": "mlx5_0", "port": 1, "pid": 10},
        {"ifname": "mlx5_0", "port": 1, "pid": 11},
        {"ifname": "mlx5_1", "port": 1, "pid": 10},
        {"ifname": "mlx5_0", "port": 1, "type": "GSI"},
    ]
    assert observe.qp_counts(entries) == Counter({"mlx5_0:1": 3, "mlx5_1:1": 1})
    assert observe.qp_counts(entries, {10}) == Counter({"mlx5_0:1": 1, "mlx5_1:1": 1})


def test_descendants(tmp_path: Path):
    for pid, ppid, comm in (
        (1, 0, "init"),
        (100, 1, "mpirun"),
        (101, 100, "a b) c"),
        (102, 101, "rank"),
        (200, 1, "x"),
    ):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0\n")
    assert observe.descendants(100, tmp_path) == {100, 101, 102}


def test_parse_nccl_tests():
    avg, rows = validate.parse(NCCL_TESTS_OUT)
    assert avg == 21.785
    assert rows == [(8, 35.11, 0.0), (1048576, 45.12, 43.57)]


def test_validate_command():
    cmd = validate.command(
        "mpirun -np 16 --bind-to none", "worst", "alltoall_perf", ["-b", "8"], ["--topo-file", "/t.xml"]
    )
    assert cmd == [
        "mpirun",
        "-np",
        "16",
        "--bind-to",
        "none",
        "railmap",
        "exec",
        "--variant",
        "worst",
        "--topo-file",
        "/t.xml",
        "--",
        "alltoall_perf",
        "-b",
        "8",
    ]
    assert validate.command("mpirun", "none", "a", [], []) == ["mpirun", "a"]
