import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from railmap.topology.discover import discover
from railmap.ui import campaign, server
from tests import fakesys

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def state(tmp_path: Path) -> server.State:
    fakesys.build_azure_vm(tmp_path / "sys")
    return server.State(discover(root=tmp_path / "sys", env={}), tmp_path / "results")


def write_campaign(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "profile.json").write_text(
        json.dumps({"name": "measured-x", "kind": "measured", "match": {}, "knee_qps": 1024, "curve": [[1, 1.0]]})
    )
    (d / "obs-q4.json").write_text(json.dumps({"peak_per_nic": {"mlx5_ib0": 130, "mlx5_ib1": 128}}))
    (d / "model-q4.json").write_text(json.dumps({"budget": {"qps_per_nic": 128}}))
    (d / "a2a-q4.txt").write_text(
        "       8   2 float none -1   4.00  0.00  0.00  0   4.10  0.00  0.00  0\n"
        "    4096 1024 float none -1  10.00  0.41  0.38  0  10.20  0.40  0.38  0\n"
    )
    (d / "obs-q16.json").write_text(json.dumps({"peak_per_nic": {}}))


def test_topology_view(state):
    v = server.topology_view(state.topo)
    assert len(v["gpus"]) == 8 and len(v["paths"]) == 8
    assert all(len(row) == len(v["nics"]) for row in v["paths"])
    assert v["paths"][0][0] == "PIX" and v["paths"][0][7] == "SYS"
    assert all(n["placement"] for n in v["nics"])
    assert v["numa_nodes"][1] == {"id": 1, "cpus": "48-95", "count": 48}


def test_plan_view(state):
    v = server.plan_view(state.topo, "best", None)
    assert [r["nics"][0] for r in v["plan"]["ranks"]] == [f"mlx5_ib{i}:1" for i in range(8)]
    assert "export UCX_NET_DEVICES=mlx5_ib5:1" in v["env"][5]
    assert v["slurm"].startswith("--cpu-bind=mask_cpu:")
    assert server.plan_view(state.topo, "best", 4)["plan"]["ranks"][-1]["local_rank"] == 3
    with pytest.raises(ValueError):
        server.plan_view(state.topo, "best", 9)


def test_budget_view_matches_cli_defaults(state):
    v = server.budget_view(state, {"nodes": "128"})
    b = v["budget"]
    assert b["qps_per_nic"] == 127 * 8 * 2 * 2 and b["status"] == "over" and b["max_nodes"] == 94
    assert v["profile"]["id"] == "shipped/prior.json"
    labels = [s["label"] for s in v["series"]]
    assert labels[0] == "this job" and labels[1].startswith("hierarchical")
    assert all(len(s["qps"]) == len(v["nodes_axis"]) for s in v["series"])
    rail = v["series"][1]
    assert rail["qps_at_job"] == 127 * 2 * 2 and rail["max_nodes"] == 750


def test_budget_view_drops_alternatives_that_change_nothing(state):
    # NVSHMEM alternatives (DC, one RC per peer) do not affect an all-reduce estimate.
    v = server.budget_view(state, {"nodes": "64", "pattern": "allreduce", "stack": "nvshmem", "channels": "4"})
    assert [s["label"] for s in v["series"]] == ["this job"]


def test_budget_view_rejects_bad_input(state):
    with pytest.raises(ValueError, match="unknown profile"):
        server.budget_view(state, {"profile": "../../etc/passwd"})
    with pytest.raises(ValueError, match="at most"):
        server.budget_view(state, {"nodes": "1000000"})
    with pytest.raises(ValueError, match="integer"):
        server.budget_view(state, {"nodes": "lots"})
    with pytest.raises(ValueError, match="pattern"):
        server.budget_view(state, {"pattern": "ring"})


def test_campaign_profiles_are_selectable(state):
    write_campaign(state.results / "azure-ndv5-20261020-1900")
    v = server.budget_view(state, {"profile": "campaign/azure-ndv5-20261020-1900"})
    assert v["budget"]["knee_qps"] == 1024 and v["profile"]["kind"] == "measured"


def test_node_axis():
    axis = server.node_axis(128)
    assert axis[0] == 2 and axis[-1] == 1024 and 128 in axis and axis == sorted(set(axis))
    assert server.node_axis(1)[0] == 2
    assert server.node_axis(50_000)[-1] == server.MAX_NODES


def test_campaign_load(tmp_path: Path):
    write_campaign(tmp_path / "azure-ndv5-20261020-1900")
    (tmp_path / "azure-ndv5-20261019-1200").mkdir()  # nothing measured yet: not listed
    (tmp_path / "azure-ndv5-20261021-0800").mkdir()
    (tmp_path / "azure-ndv5-20261021-0800" / "plan.json").write_text("{}")
    assert campaign.list_campaigns(tmp_path) == ["azure-ndv5-20261021-0800", "azure-ndv5-20261020-1900"]
    assert campaign.list_campaigns(tmp_path / "missing") == []
    d = campaign.load(tmp_path / "azure-ndv5-20261020-1900")
    assert d["profile"]["knee_qps"] == 1024 and d["gdr"] is None and d["validate"] is None
    assert d["inflation"] == [
        {"qps_per_conn": 4, "model": 128, "observed": 130, "small_time_us": 4.0},
        {"qps_per_conn": 16, "model": None, "observed": 0, "small_time_us": None},
    ]


def test_load_upload_snapshot_and_xml(state):
    snap = server.load_upload(json.dumps(state.topo.to_dict()).encode(), "/some/dir/node0.json")
    assert snap.source == "upload:node0.json" and len(snap.gpus) == 8
    xml = server.load_upload((FIXTURES / "azure-ndv5-topo.xml").read_bytes(), "ndv5-topo.xml")
    assert xml.topo_file is None and len(xml.nics) == 8
    assert any("--topo-file" in n for n in xml.notes)
    with pytest.raises(ValueError):
        server.load_upload(b"not json", "x.json")


@pytest.fixture
def http(state):
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(url: str, data: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_routes(http, state):
    status, page = call(f"{http}/")
    assert status == 200 and b"<title>railmap</title>" in page
    status, body = call(f"{http}/api/topology")
    assert status == 200 and json.loads(body)["is_host"]
    status, body = call(f"{http}/api/plan?ranks=abc")
    assert status == 400 and "integer" in json.loads(body)["error"]
    assert call(f"{http}/api/campaign?name=..")[0] == 400
    assert call(f"{http}/nope")[0] == 404


def test_http_refuses_foreign_host(http):
    assert call(f"{http}/api/topology", headers={"Host": "attacker.example:8765"})[0] == 403
    assert call(f"{http}/api/topology", headers={"Host": "localhost:9000"})[0] == 200  # ssh -L to another port


def test_http_upload_and_reset(http, state):
    xml = (FIXTURES / "azure-ndv5-topo.xml").read_bytes()
    status, _ = call(f"{http}/api/topology", xml, {"X-Filename": "ndv5%20topo.xml"})
    assert status == 200 and state.topo.source == "upload:ndv5 topo.xml"
    assert not json.loads(call(f"{http}/api/topology")[1])["is_host"]
    assert call(f"{http}/api/topology", b"{}", {"X-Filename": "bad.json"})[0] == 400
    assert call(f"{http}/api/topology/reset", b"")[0] == 200
    assert state.topo is state.host
