import json
import resource
from pathlib import Path

from typer.testing import CliRunner

from railmap import doctor
from railmap.cli import app
from tests import fakesys


def none(_: str) -> None:
    return None


def by_name(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    return {c.name: c for c in checks}


def test_empty_host_reports_without_raising(tmp_path: Path):
    checks = by_name(doctor.run(root=tmp_path, env={}, which=none, memlock=resource.RLIM_INFINITY))
    assert checks["GPUs"].status == "fail"
    assert checks["RDMA NICs"].status == "warn"
    assert checks["numactl"].status == "ok"  # one NUMA node: nothing to bind
    assert checks["perftest"].status == "warn"


def test_azure_container_without_devices(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path, with_topo_file=False)
    fakesys.add_uverbs(tmp_path, [f"mlx5_ib{i}" for i in range(8)], present=set())
    (tmp_path / ".dockerenv").touch()
    checks = by_name(doctor.run(root=tmp_path, env={}, which=none, memlock=8 << 20))
    assert checks["container"].detail == "Docker container"
    assert checks["verbs devices"].status == "fail" and "--device /dev/infiniband" in checks["verbs devices"].fix
    assert checks["topology"].status == "ok"  # the bundled ndv5 file matched
    assert checks["locked memory"].status == "warn" and "8 MiB" in checks["locked memory"].detail
    assert checks["numactl"].status == "warn"  # two NUMA nodes


def test_cgroup_without_cpus_near_gpus(tmp_path: Path):
    fakesys.build_azure_vm(tmp_path)
    fakesys.add_cgroup_v2(tmp_path, "0-11", rel="/kubepods/x")
    fakesys.add_mems(tmp_path, "0")
    env = {"KUBERNETES_SERVICE_HOST": "10.0.0.1", "LOCAL_RANK": "3"}
    checks = by_name(doctor.run(root=tmp_path, env=env, which=none, memlock=resource.RLIM_INFINITY))
    assert checks["container"].detail == "Kubernetes pod"
    assert checks["CPUs"].status == "warn" and "12 of" in checks["CPUs"].detail
    assert checks["NUMA memory"].status == "warn"
    assert checks["launcher"].detail == "LOCAL_RANK=3"


def test_tools_present():
    checks = by_name(doctor.check_tools(lambda t: f"/usr/bin/{t}", 2, cuda=lambda _: True))
    assert all(checks[t].status == "ok" for t in ("numactl", "rdma (iproute2)", "perftest"))
    checks = by_name(doctor.check_tools(lambda t: f"/usr/bin/{t}", 2, cuda=lambda _: False))
    assert checks["perftest"].status == "warn"


def test_memlock_thresholds():
    assert doctor.check_memlock(resource.RLIM_INFINITY).status == "ok"
    assert doctor.check_memlock(128 << 30).status == "ok"
    assert doctor.check_memlock(64 << 10).status == "warn"


def test_cli_doctor_json_and_exit_code(monkeypatch):
    monkeypatch.setattr(doctor, "run", lambda **_: [doctor.Check("x", "fail", "broken", "fix it")])
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == [{"name": "x", "status": "fail", "detail": "broken", "fix": "fix it"}]


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0 and result.stdout.startswith("railmap ")
