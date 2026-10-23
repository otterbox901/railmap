from pathlib import Path

from railmap.topology import nccl_xml, sysfs
from railmap.topology.discover import BUNDLED_TOPO_DIR, discover, placement_nics
from railmap.topology.model import PathClass, Topology, classify, format_cpulist, parse_cpulist
from tests import fakesys

AZURE_XML = fakesys.FIXTURES / "azure-ndv5-topo.xml"


def test_cpulist_roundtrip():
    assert parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert format_cpulist([11, 0, 1, 2, 3, 8, 10]) == "0-3,8,10-11"


def test_classify_sysfs_switch_levels():
    root = ["pci0000:00", "0000:00:01.0", "0000:01:00.0"]
    assert classify(root + ["0000:02:00.0"], root + ["0000:02:01.0"], 0, 0) is PathClass.PIX
    nested = root + ["0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]
    assert classify(nested, root + ["0000:02:01.0"], 0, 0) is PathClass.PXB
    assert classify(["pci0000:00", "0000:00:01.0"], ["pci0000:00", "0000:00:1c.0"], 0, 0) is PathClass.PHB
    assert classify(["pci0000:00"], ["pci0000:80"], 0, 0) is PathClass.NODE
    assert classify(["pci0000:00"], ["pci0000:80"], 0, 1) is PathClass.SYS


def test_azure_xml_paths():
    topo = nccl_xml.parse(AZURE_XML)
    assert [g.bdf for g in topo.gpus] == [f"{d}:00:00.0" for d in fakesys.AZURE_GPUS]
    assert len(topo.nics) == 8
    assert [n.id for n in topo.numa_nodes] == [0, 1]
    assert topo.numa_nodes[0].cpus == list(range(48))
    assert topo.numa_nodes[1].cpus == list(range(48, 96))
    for i, g in enumerate(topo.gpus):
        for j, n in enumerate(topo.nics):
            want = PathClass.PIX if i == j else PathClass.PHB if i // 4 == j // 4 else PathClass.SYS
            assert topo.path(g, n) is want, (i, j)


def test_dgx_sysfs_paths(tmp_path: Path):
    fakesys.build_dgx(tmp_path)
    topo = sysfs.scan(tmp_path)
    assert len(topo.gpus) == 8 and len(topo.nics) == 8
    assert not topo.flattened
    g0 = topo.gpus[0]
    classes = [topo.path(g0, n) for n in topo.nics]
    assert classes == [PathClass.PIX, PathClass.PXB, PathClass.NODE, PathClass.NODE] + [PathClass.SYS] * 4
    nic = topo.nics[0]
    assert (nic.name, nic.link_layer, nic.rate_gbps, nic.state) == ("mlx5_0", "InfiniBand", 400.0, "ACTIVE")


def test_flattened_vm_uses_vendor_file(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    topo = discover(root=tmp_path, env={})
    assert topo.flattened and topo.topo_file == str(tmp_path / "opt/microsoft/ndv5-topo.xml")
    assert topo.warnings() == []
    assert [n.name for n in topo.nics] == [f"mlx5_ib{i}" for i in range(8)]
    assert any("mlx5_an0" in n for n in topo.notes)
    assert all(topo.path(g, topo.nics[g.index]) is PathClass.PIX for g in topo.gpus)


def test_flattened_vm_without_file_warns(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path, with_topo_file=False)
    topo = discover(root=tmp_path, env={}, bundled_dir=None)
    assert topo.flattened and topo.topo_file is None
    assert topo.warnings()
    # Without the file every GPU-NIC pair looks equally far, which is the failure this guards against.
    assert {topo.path(topo.gpus[0], n) for n in topo.nics if n.numa == 0} == {PathClass.NODE}


def test_flattened_vm_falls_back_to_bundled_copy(tmp_path: Path):
    # A container on an ND H100 v5 sees the host's sysfs but not its /opt/microsoft.
    fakesys.build_azure_vm(tmp_path, with_topo_file=False)
    topo = discover(root=tmp_path, env={})
    assert topo.topo_file == str(BUNDLED_TOPO_DIR / "ndv5-topo.xml")
    assert topo.warnings() == []
    assert any("railmap's copy of ndv5-topo.xml" in n for n in topo.notes)
    assert all(topo.path(g, topo.nics[g.index]) is PathClass.PIX for g in topo.gpus)


def test_host_vendor_file_beats_bundled_copy(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    assert discover(root=tmp_path, env={}).topo_file == str(tmp_path / "opt/microsoft/ndv5-topo.xml")


def test_bundled_files_match_upstream_layouts():
    assert nccl_xml.parse(BUNDLED_TOPO_DIR / "ndv5-topo.xml").gpus == nccl_xml.parse(AZURE_XML).gpus
    ndv4 = nccl_xml.parse(BUNDLED_TOPO_DIR / "ndv4-topo.xml")
    assert len(ndv4.gpus) == 8 and len(ndv4.nics) == 8 and [n.id for n in ndv4.numa_nodes] == [0, 1, 2, 3]


def test_env_topo_file_wins(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path, with_topo_file=False)
    topo = discover(root=tmp_path, env={"NCCL_TOPO_FILE": str(AZURE_XML)})
    assert topo.topo_file == str(AZURE_XML)


def test_placement_nics_drops_slow_frontend(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path, with_topo_file=False)
    names = {n.name for n in placement_nics(sysfs.scan(tmp_path))}
    assert "mlx5_an0" not in names and len(names) == 8


def test_laptop_snapshot_loads():
    topo = Topology.load(fakesys.FIXTURES / "laptop.json")
    assert len(topo.gpus) == 1 and topo.nics == []


def test_snapshot_roundtrip(tmp_path: Path):
    topo = nccl_xml.parse(AZURE_XML)
    topo.smt_siblings = {0: [0, 1], 1: [0, 1]}
    topo.save(tmp_path / "t.json")
    back = Topology.load(tmp_path / "t.json")
    assert back.gpus == topo.gpus and back.nics == topo.nics and back.smt_siblings == topo.smt_siblings
