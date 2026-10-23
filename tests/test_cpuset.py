from pathlib import Path

from railmap.emit import execwrap
from railmap.placement.assign import assign
from railmap.topology import cpuset
from railmap.topology.discover import discover, keep_usable
from railmap.topology.model import format_cpulist
from tests import fakesys

AZURE_XML = fakesys.FIXTURES / "azure-ndv5-topo.xml"


def test_cgroup_v2(tmp_path: Path):
    fakesys.add_cgroup_v2(tmp_path, "0-3,8", rel="/kubepods/pod1/c1")
    assert cpuset.cgroup_cpus(tmp_path) == [0, 1, 2, 3, 8]


def test_cgroup_v1(tmp_path: Path):
    fakesys.add_cgroup_v1(tmp_path, "4-7", rel="/slurm/uid_1/job_2/step_0")
    assert cpuset.cgroup_cpus(tmp_path) == [4, 5, 6, 7]


def test_cgroup_path_missing_walks_up_to_mount(tmp_path: Path):
    # cgroup v1 Docker without a cgroup namespace: /proc names a host path, the container's own cgroup is the mount.
    fakesys.add_cgroup_v2(tmp_path, "0-1", rel="/docker/abc123", at="/")
    assert cpuset.cgroup_cpus(tmp_path) == [0, 1]


def test_no_cgroup_is_unknown(tmp_path: Path):
    assert cpuset.cgroup_cpus(tmp_path) is None
    assert cpuset.allowed_cpus(tmp_path) is None  # no affinity fallback for a fake root


def test_mems(tmp_path: Path):
    fakesys.add_mems(tmp_path, "0")
    assert cpuset.allowed_mems(tmp_path) == {0}
    assert cpuset.allowed_mems(tmp_path / "nowhere") is None


def test_plan_uses_only_cgroup_cpus(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    fakesys.add_cgroup_v2(tmp_path, "0-23,48-71", rel="/kubepods/pod1/c1")
    topo = discover(root=tmp_path, env={})
    assert any("48 of 96 CPUs allowed by the cgroup" in n for n in topo.notes)
    plan = assign(topo)
    allowed = set(range(24)) | set(range(48, 72))
    assert all(set(r.cpus) <= allowed for r in plan.ranks)
    assert [len(r.cpus) for r in plan.ranks] == [6] * 8  # still an even split, just of the allowed CPUs
    assert len({c for r in plan.ranks for c in r.cpus}) == 48


def test_all_cpus_and_snapshot_are_unrestricted(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    fakesys.add_cgroup_v2(tmp_path, "0-23")
    assert sum(len(n.cpus) for n in discover(root=tmp_path, env={}, all_cpus=True).numa_nodes) == 96
    snap = tmp_path / "snap.json"
    discover(root=tmp_path, env={}, all_cpus=True).save(snap)
    assert sum(len(n.cpus) for n in discover(snapshot=snap, root=tmp_path).numa_nodes) == 96


def test_foreign_topology_file_is_unrestricted(tmp_path: Path):
    # Planning on this laptop for an ND H100 v5: the file doesn't describe this host, so its cgroup doesn't apply.
    fakesys.build_dgx(tmp_path)
    fakesys.add_cgroup_v2(tmp_path, "0-3")
    topo = discover(topo_file=AZURE_XML, root=tmp_path, env={})
    assert sum(len(n.cpus) for n in topo.numa_nodes) == 96


def test_numa_node_with_no_allowed_cpus(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    fakesys.add_cgroup_v2(tmp_path, "0-47")
    plan = assign(discover(root=tmp_path, env={}))
    assert any("NUMA node 1: no usable CPUs" in w for w in plan.warnings)
    assert all(r.cpus and max(r.cpus) < 48 for r in plan.ranks)


def test_smt_siblings_split_by_cpuset(tmp_path: Path):
    fakesys.build_trap(tmp_path)
    for c in range(64):
        d = tmp_path / f"sys/devices/system/cpu/cpu{c}/topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(f"{c % 32},{c % 32 + 32}\n" if c % 32 < 16 else f"{c}\n")
    fakesys.add_cgroup_v2(tmp_path, "0-15,32-39")
    plan = assign(discover(root=tmp_path, env={}))
    assert all(set(r.cpus) <= set(range(16)) | set(range(32, 40)) for r in plan.ranks)


def test_exec_skips_membind_outside_mems(tmp_path: Path, monkeypatch):
    fakesys.build_azure_vm(tmp_path)
    plan = assign(discover(root=tmp_path, env={}))
    monkeypatch.setattr(execwrap.shutil, "which", lambda _: "/usr/bin/numactl")
    argv, _, _ = execwrap.build(plan, ["train"], 5, {}, False, mems={0})
    assert argv == ["train"]
    argv, _, _ = execwrap.build(plan, ["train"], 5, {}, False, mems={0, 1})
    assert argv == ["numactl", "--membind=1", "train"]


def test_keep_usable_gpus_renumbers(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    topo = discover(root=tmp_path, env={})
    mine = {g.bdf for g in topo.gpus[4:]}
    keep_usable(topo, mine, {})
    assert [g.index for g in topo.gpus] == [0, 1, 2, 3] and {g.bdf for g in topo.gpus} == mine
    plan = assign(topo)
    assert [r.nics[0] for r in plan.ranks] == [f"mlx5_ib{i}:1" for i in range(4, 8)]  # still each GPU's own NIC
    assert all(min(r.cpus) >= 48 for r in plan.ranks)


def test_keep_usable_ignores_all_or_nothing(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    topo = discover(root=tmp_path, env={})
    keep_usable(topo, set(), {n.name: False for n in topo.nics})
    assert len(topo.gpus) == 8 and len(topo.nics) == 8


def test_container_with_some_verbs_devices(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    names = [f"mlx5_ib{i}" for i in range(8)] + ["mlx5_an0"]
    fakesys.add_uverbs(tmp_path, names, present={"mlx5_ib0", "mlx5_ib1"})
    topo = discover(root=tmp_path, env={})
    assert [n.name for n in topo.nics] == ["mlx5_ib0", "mlx5_ib1"]
    assert any("no /dev/infiniband node" in n for n in topo.notes)
    plan = assign(topo, ranks=2)
    assert format_cpulist(plan.ranks[0].cpus) == "0-23"
