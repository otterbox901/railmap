from collections import Counter
from pathlib import Path

import pytest

from railmap.placement.assign import Plan, assign
from railmap.topology import nccl_xml, sysfs
from railmap.topology.discover import discover
from tests import fakesys

AZURE_XML = fakesys.FIXTURES / "azure-ndv5-topo.xml"


def test_azure_pairs_each_gpu_with_its_pix_nic():
    plan = assign(nccl_xml.parse(AZURE_XML))
    assert [r.path for r in plan.ranks] == ["PIX"] * 8
    assert [r.nics for r in plan.ranks] == [[f"pci-01{i + 1:02x}:00:00.0:1"] for i in range(8)]
    assert [r.numa for r in plan.ranks] == [0] * 4 + [1] * 4
    assert [r.cpus for r in plan.ranks[:2]] == [list(range(12)), list(range(12, 24))]
    assert plan.warnings == []


def test_trap_avoids_sys(tmp_path: Path):
    fakesys.build_trap(tmp_path)
    topo = sysfs.scan(tmp_path)
    plan = assign(topo)
    assert {r.path for r in plan.ranks} == {"PIX"}
    assert Counter(r.nics[0] for r in plan.ranks) == {f"mlx5_{i}:1": 2 for i in range(4)}
    assert [r.nics[0] for r in plan.ranks[:2]] == ["mlx5_2:1", "mlx5_2:1"]
    assert plan.warnings == []


def test_worst_variant_crosses_sockets():
    plan = assign(nccl_xml.parse(AZURE_XML), variant="worst")
    assert {r.path for r in plan.ranks} == {"SYS"}
    assert [r.numa for r in plan.ranks] == [1] * 4 + [0] * 4


def test_more_nics_than_gpus_shards(tmp_path: Path):
    fakesys.build_4x8(tmp_path)
    plan = assign(sysfs.scan(tmp_path))
    assert [r.nics for r in plan.ranks] == [[f"mlx5_{2 * i}:1", f"mlx5_{2 * i + 1}:1"] for i in range(4)]
    assert {r.path for r in plan.ranks} == {"PIX"}


def test_plan_identical_across_nodes(tmp_path: Path):
    plans = []
    for node in ("a", "b"):
        fakesys.build_dgx(tmp_path / node)
        p = assign(sysfs.scan(tmp_path / node))
        plans.append([(r.gpu, r.nics, r.cpus) for r in p.ranks])
    assert plans[0] == plans[1]


def test_cpus_stay_in_numa_node_and_keep_smt_siblings(tmp_path: Path):
    fakesys.build_dgx(tmp_path)
    topo = sysfs.scan(tmp_path)
    # 56 cores per socket with hyperthread siblings at +112
    for node in topo.numa_nodes:
        node.cpus = node.cpus + [c + 112 for c in node.cpus]
    topo.smt_siblings = {c: [c % 112, c % 112 + 112] for n in topo.numa_nodes for c in n.cpus}
    plan = assign(topo)
    for r in plan.ranks:
        allowed = set(topo.cpus_of(r.numa))
        assert set(r.cpus) <= allowed
        assert all((c + 112 if c < 112 else c - 112) in r.cpus for c in r.cpus)
    assert len(plan.ranks[0].cpus) == 28


def test_flattened_without_file_warns(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path, with_topo_file=False)
    plan = assign(discover(root=tmp_path, env={}))
    assert any("virtualized" in w for w in plan.warnings)


def test_plan_roundtrip(tmp_path: Path):
    plan = assign(nccl_xml.parse(AZURE_XML))
    plan.save(tmp_path / "plan.json")
    assert Plan.load(tmp_path / "plan.json") == plan


def test_too_many_ranks():
    with pytest.raises(ValueError):
        assign(nccl_xml.parse(AZURE_XML), ranks=9)
