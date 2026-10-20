from pathlib import Path

import pytest

from railmap.emit import env, execwrap
from railmap.placement.assign import assign
from railmap.topology import nccl_xml
from railmap.topology.discover import discover
from tests import fakesys


@pytest.fixture
def azure_plan(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    return assign(discover(root=tmp_path, env={}))


def test_rank_env(azure_plan):
    e = env.rank_env(azure_plan, 5)
    assert e["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert e["NCCL_TOPO_FILE"].endswith("opt/microsoft/ndv5-topo.xml")
    assert e["NCCL_IB_HCA"] == "=" + ",".join(f"mlx5_ib{i}:1" for i in range(8))
    assert e["UCX_NET_DEVICES"] == "mlx5_ib5:1"
    assert e["NVSHMEM_HCA_PE_MAPPING"] == ",".join(f"mlx5_ib{i}:1:1" for i in range(8))
    assert e["RAILMAP_CPUS"] == "60-71" and e["RAILMAP_NUMA"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in e
    assert env.rank_env(azure_plan, 5, isolate_gpu=True)["CUDA_VISIBLE_DEVICES"] == "5"


def test_worst_env_pins_far_nic(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    plan = assign(discover(root=tmp_path, env={}), variant="worst")
    e = env.rank_env(plan, 0)
    assert e["NCCL_IB_HCA"].startswith("=mlx5_ib") and "," not in e["NCCL_IB_HCA"]
    assert e["RAILMAP_NUMA"] == "1"


def test_slurm_mask(azure_plan):
    flag = env.slurm_cpu_bind(azure_plan)
    masks = flag.removeprefix("--cpu-bind=mask_cpu:").split(",")
    assert masks[0] == hex((1 << 12) - 1) and len(masks) == 8


def test_shell_quoting():
    assert env.shell_exports({"A": "x y", "B": "=a:1,b:1"}) == "export A='x y'\nexport B==a:1,b:1"


def test_detect_local_rank():
    assert execwrap.detect_local_rank({"OMPI_COMM_WORLD_LOCAL_RANK": "3"}) == 3
    assert execwrap.detect_local_rank({"SLURM_LOCALID": "1", "LOCAL_RANK": "2"}) == 2
    with pytest.raises(RuntimeError):
        execwrap.detect_local_rank({})


def test_exec_build_membind(azure_plan, monkeypatch):
    monkeypatch.setattr(execwrap.shutil, "which", lambda _: "/usr/bin/numactl")
    argv, e, cpus = execwrap.build(azure_plan, ["python", "train.py"], 6, {"PATH": "/usr/bin"}, False)
    assert argv == ["numactl", "--membind=1", "python", "train.py"]
    assert e["PATH"] == "/usr/bin" and e["UCX_NET_DEVICES"] == "mlx5_ib6:1"
    assert cpus == list(range(72, 84))
    monkeypatch.setattr(execwrap.shutil, "which", lambda _: None)
    argv, _, _ = execwrap.build(azure_plan, ["python"], 6, {}, False)
    assert argv == ["python"]


def test_offline_plan_omits_nic_vars():
    plan = assign(nccl_xml.parse(fakesys.FIXTURES / "azure-ndv5-topo.xml"))
    e = env.rank_env(plan, 0)
    assert "NCCL_IB_HCA" not in e and "UCX_NET_DEVICES" not in e
    assert e["NCCL_TOPO_FILE"].startswith("/")
    assert any("unknown offline" in n for n in plan.notes)
