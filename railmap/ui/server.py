"""`railmap ui`: a local web page for the topology, placement, connection budget and campaign results."""

from __future__ import annotations

import json
import tempfile
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from railmap.emit import env as envmod
from railmap.placement import budget as budgetmod
from railmap.placement.assign import assign
from railmap.placement.connections import Job, estimate, max_nodes_under
from railmap.topology import nccl_xml
from railmap.topology.discover import placement_nics
from railmap.topology.model import Topology, format_cpulist
from railmap.ui import campaign

MAX_NODES = 100_000
MAX_UPLOAD = 8 << 20
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class State:
    """The topology the page plans on (the host's, or one the user loaded) and where results live."""

    def __init__(self, topo: Topology, results: Path, profile: Path | None = None) -> None:
        self.host = topo
        self.topo = topo
        self.results = results
        self.profile = profile

    def profiles(self) -> dict[str, Path]:
        """Selectable NIC profiles by id: shipped ones, --profile, and each campaign's measured profile."""
        out = {f"shipped/{p.name}": p for p in sorted(budgetmod.PROFILE_DIR.glob("*.json"))}
        if self.profile:
            out[f"file/{self.profile.name}"] = self.profile
        for name in campaign.list_campaigns(self.results):
            if (self.results / name / "profile.json").exists():
                out[f"campaign/{name}"] = self.results / name / "profile.json"
        return out


def _int(q: dict[str, str], key: str, default: int | None) -> int | None:
    if not q.get(key):
        return default
    try:
        return int(q[key])
    except ValueError:
        raise ValueError(f"{key} must be an integer") from None


def topology_view(topo: Topology) -> dict:
    usable = {n.label for n in placement_nics(topo)}
    return {
        "source": topo.source,
        "topo_file": topo.topo_file,
        "warnings": topo.warnings(),
        "notes": topo.notes,
        "gpus": [asdict(g) for g in topo.gpus],
        "nics": [asdict(n) | {"label": n.label, "placement": n.label in usable} for n in topo.nics],
        "numa_nodes": [{"id": n.id, "cpus": format_cpulist(n.cpus), "count": len(n.cpus)} for n in topo.numa_nodes],
        "paths": [[topo.path(g, n).name for n in topo.nics] for g in topo.gpus],
    }


def plan_view(topo: Topology, variant: str, ranks: int | None) -> dict:
    p = assign(topo, ranks=ranks, variant=variant)
    return {
        "plan": asdict(p),
        "env": {r.local_rank: envmod.shell_exports(envmod.rank_env(p, r.local_rank)) for r in p.ranks},
        "slurm": envmod.slurm_cpu_bind(p),
    }


def node_axis(nodes: int, points: int = 48) -> list[int]:
    """Log-spaced node counts from 2 to well past the job size, plus the job size itself."""
    hi = min(MAX_NODES, max(1024, 4 * nodes))
    axis = {round(2 * (hi / 2) ** (i / (points - 1))) for i in range(points)}
    return sorted(axis | ({nodes} if nodes >= 2 else set()))


def budget_view(state: State, q: dict[str, str]) -> dict:
    topo = state.topo
    nics = placement_nics(topo)
    profiles = state.profiles()
    default = budgetmod.select_profile(nics[0] if nics else None, state.profile)
    pid = q.get("profile") or next(k for k, p in profiles.items() if str(p) == default.path)
    if pid not in profiles:
        raise ValueError(f"unknown profile {pid!r}")
    profile = budgetmod.load_profile(profiles[pid])
    job = Job(
        nodes=_int(q, "nodes", 128),
        gpus_per_node=_int(q, "gpus_per_node", len(topo.gpus) or 8),
        nics_per_node=_int(q, "nics_per_node", max(1, len(nics))),
        pattern=q.get("pattern", "alltoall"),
        stack=q.get("stack", "nccl"),
        channels_per_peer=_int(q, "channels_per_peer", 2),
        qps_per_conn=_int(q, "qps_per_conn", 1),
        channels=_int(q, "channels", 16),
        rc_per_peer=_int(q, "rc_per_peer", 2),
        dc=q.get("dc") == "1",
        rail_only=q.get("rail_only") == "1",
    )
    if job.nodes > MAX_NODES:
        raise ValueError(f"nodes must be at most {MAX_NODES}")

    axis = node_axis(job.nodes)
    current = [estimate(replace(job, nodes=n)).qps_per_nic for n in axis]
    series = [{"label": "this job", "setting": "", "qps": current, "qps_at_job": estimate(job).qps_per_nic}]
    for change, setting, alt in budgetmod.alternatives(job):
        qps = [estimate(replace(alt, nodes=n)).qps_per_nic for n in axis]
        if qps != current:
            series.append(
                {
                    "label": change,
                    "setting": setting,
                    "qps": qps,
                    "qps_at_job": estimate(alt).qps_per_nic,
                    "max_nodes": max_nodes_under(alt, profile.knee_qps),
                }
            )
    return {
        "budget": asdict(budgetmod.evaluate(job, profile)),
        "profile": {"id": pid} | {k: getattr(profile, k) for k in ("name", "kind", "knee_qps", "curve", "source")},
        "profiles": [{"id": k, "name": budgetmod.load_profile(p).name} for k, p in profiles.items()],
        "nodes_axis": axis,
        "series": series,
    }


def campaign_view(state: State, name: str) -> dict:
    if name not in campaign.list_campaigns(state.results):
        raise ValueError(f"no campaign {name!r} in {state.results}")
    return campaign.load(state.results / name)


def load_upload(body: bytes, filename: str) -> Topology:
    """A topology from `railmap show --save` JSON or an NCCL topology XML, for planning offline."""
    text = body.decode()
    name = Path(filename).name or "upload"
    if text.lstrip().startswith("<"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topo.xml"
            path.write_text(text)
            topo = nccl_xml.parse(path)
        # The temporary path means nothing on the target host, so keep it out of NCCL_TOPO_FILE.
        topo.topo_file = None
        topo.notes.append(f"from NCCL topology XML {name}: on the target host, pass it with --topo-file")
    else:
        topo = Topology.from_dict(json.loads(text))
    topo.source = f"upload:{name}"
    return topo


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass

        def _send(self, status: HTTPStatus, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, fn: Callable[[], dict]) -> None:
            try:
                body, status = fn(), HTTPStatus.OK
            except (ValueError, KeyError, TypeError) as e:
                body, status = {"error": str(e)}, HTTPStatus.BAD_REQUEST
            self._send(status, json.dumps(body).encode(), "application/json")

        def _local(self) -> bool:
            # Refuse other Host names so a web page cannot reach this server through DNS rebinding.
            host = urlparse(f"//{self.headers.get('Host', '')}").hostname or ""
            if host in LOCAL_HOSTS or host == self.server.server_address[0]:
                return True
            self._send(HTTPStatus.FORBIDDEN, b"forbidden host", "text/plain")
            return False

        def do_GET(self) -> None:
            if not self._local():
                return
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            routes: dict[str, Callable[[], dict]] = {
                "/api/topology": lambda: topology_view(state.topo) | {"is_host": state.topo is state.host},
                "/api/plan": lambda: plan_view(state.topo, q.get("variant", "best"), _int(q, "ranks", None)),
                "/api/budget": lambda: budget_view(state, q),
                "/api/campaigns": lambda: {
                    "results": str(state.results.resolve()),
                    "campaigns": campaign.list_campaigns(state.results),
                },
                "/api/campaign": lambda: campaign_view(state, q.get("name", "")),
            }
            if url.path in ("/", "/index.html"):
                page = resources.files("railmap.ui").joinpath("index.html").read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif url.path in routes:
                self._json(routes[url.path])
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            if not self._local():
                return
            path = urlparse(self.path).path
            if path == "/api/topology/reset":
                state.topo = state.host
                self._json(lambda: {"ok": True})
            elif path == "/api/topology":
                length = int(self.headers.get("Content-Length", 0))
                if length > MAX_UPLOAD:
                    self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"topology file too large", "text/plain")
                    return
                body = self.rfile.read(length)

                def upload() -> dict:
                    state.topo = load_upload(body, unquote(self.headers.get("X-Filename", "")))
                    return {"ok": True}

                self._json(upload)
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    return Handler


def serve(state: State, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    shown = "localhost" if host in LOCAL_HOSTS else host
    url = f"http://{shown}:{server.server_address[1]}/"
    print(f"railmap ui: {url}  (Ctrl-C to stop)", flush=True)
    if host in LOCAL_HOSTS:
        print(
            f"from another machine: ssh -L {server.server_address[1]}:localhost:{server.server_address[1]} <node>",
            flush=True,
        )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
