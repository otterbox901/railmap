"""railmap command line."""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from railmap.emit import env as envmod
from railmap.emit import execwrap, report
from railmap.placement import budget as budgetmod
from railmap.placement.assign import Plan, assign
from railmap.placement.connections import Job
from railmap.probe import calibrate as cal
from railmap.probe import gdr_matrix as gm
from railmap.probe import observe as obsmod
from railmap.probe import perftest
from railmap.probe import validate as val
from railmap.topology.discover import discover, placement_nics
from railmap.topology.model import Topology
from railmap.ui import server as uiserver

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
console = Console()

TopoFile = Annotated[
    Path | None, typer.Option("--topo-file", help="NCCL topology XML (e.g. /opt/microsoft/ndv5-topo.xml)")
]
Snapshot = Annotated[Path | None, typer.Option("--snapshot", help="topology JSON saved with `show --save`")]
PlanFile = Annotated[Path | None, typer.Option("--plan", help="plan JSON from `plan -o`; computed live if omitted")]


def _topology(topo_file: Path | None, snapshot: Path | None) -> Topology:
    return discover(topo_file=topo_file, snapshot=snapshot)


def _plan(plan_file: Path | None, topo_file: Path | None, snapshot: Path | None, variant: str = "best") -> Plan:
    if plan_file:
        return Plan.load(plan_file)
    return assign(_topology(topo_file, snapshot), variant=variant)


@app.command()
def show(
    topo_file: TopoFile = None,
    snapshot: Snapshot = None,
    save: Annotated[Path | None, typer.Option(help="write the topology as JSON")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="print JSON instead of tables")] = False,
) -> None:
    """Show GPUs, RDMA NICs, NUMA nodes and the PCI path class between each GPU and NIC."""
    topo = _topology(topo_file, snapshot)
    if save:
        topo.save(save)
    if as_json:
        print(json.dumps(topo.to_dict(), indent=2))
    else:
        report.show_topology(topo, console)


@app.command()
def plan(
    topo_file: TopoFile = None,
    snapshot: Snapshot = None,
    ranks: Annotated[int | None, typer.Option(help="local ranks (default: one per GPU)")] = None,
    variant: Annotated[str, typer.Option(help="best, or worst (deliberately bad, for validation)")] = "best",
    output: Annotated[Path | None, typer.Option("-o", "--output", help="write the plan as JSON")] = None,
    nodes: Annotated[int | None, typer.Option(help="job size in nodes; enables the connection budget")] = None,
    pattern: Annotated[str, typer.Option(help="alltoall or allreduce")] = "alltoall",
    stack: Annotated[str, typer.Option(help="nccl or nvshmem")] = "nccl",
    nics_per_node: Annotated[int | None, typer.Option(help="default: NICs in this plan")] = None,
    channels_per_peer: Annotated[int, typer.Option(help="NCCL network channels per peer")] = 2,
    qps_per_conn: Annotated[int, typer.Option(help="NCCL_IB_QPS_PER_CONNECTION")] = 1,
    channels: Annotated[int, typer.Option(help="NCCL channels for allreduce")] = 16,
    rc_per_peer: Annotated[int, typer.Option(help="NVSHMEM RC QPs per PE per peer")] = 2,
    dc: Annotated[bool, typer.Option(help="NVSHMEM uses DC transport")] = False,
    rail_only: Annotated[bool, typer.Option(help="hierarchical all-to-all (same-rail RDMA + NVLink)")] = False,
    profile: Annotated[Path | None, typer.Option(help="NIC calibration profile JSON")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="print JSON instead of tables")] = False,
) -> None:
    """Assign each local rank a GPU, NIC(s), CPUs and NUMA node; with --nodes, check the QP budget."""
    topo = _topology(topo_file, snapshot)
    p = assign(topo, ranks=ranks, variant=variant)
    if output:
        p.save(output)
    result: dict = {"plan": asdict(p)}
    b = None
    if nodes:
        nics = placement_nics(topo)
        job = Job(
            nodes=nodes,
            gpus_per_node=len(p.ranks),
            nics_per_node=nics_per_node or max(1, len(nics)),
            pattern=pattern,
            stack=stack,
            channels_per_peer=channels_per_peer,
            qps_per_conn=qps_per_conn,
            channels=channels,
            rc_per_peer=rc_per_peer,
            dc=dc,
            rail_only=rail_only,
        )
        b = budgetmod.evaluate(job, budgetmod.select_profile(nics[0] if nics else None, profile))
        result["budget"] = asdict(b)
    if as_json:
        print(json.dumps(result, indent=2))
        return
    report.show_plan(p, console)
    if b:
        console.print()
        report.show_budget(b, console)


@app.command()
def env(
    local_rank: Annotated[int, typer.Option(help="local rank to print")],
    plan_file: PlanFile = None,
    topo_file: TopoFile = None,
    snapshot: Snapshot = None,
    fmt: Annotated[str, typer.Option("--format", help="sh (exports) or slurm (srun --cpu-bind flag)")] = "sh",
    isolate_gpu: Annotated[bool, typer.Option(help="also set CUDA_VISIBLE_DEVICES to the rank's GPU")] = False,
) -> None:
    """Print the environment for one local rank, or the srun CPU binding for all ranks."""
    p = _plan(plan_file, topo_file, snapshot)
    if fmt == "slurm":
        print(envmod.slurm_cpu_bind(p))
    elif fmt == "sh":
        print(envmod.shell_exports(envmod.rank_env(p, local_rank, isolate_gpu)))
    else:
        raise typer.BadParameter("--format must be sh or slurm")


@app.command("exec")
def exec_(
    cmd: Annotated[list[str], typer.Argument(help="command to run, after --")],
    plan_file: PlanFile = None,
    topo_file: TopoFile = None,
    snapshot: Snapshot = None,
    variant: Annotated[str, typer.Option(help="used when no --plan is given")] = "best",
    local_rank: Annotated[int | None, typer.Option(help="default: from LOCAL_RANK, OMPI_..., SLURM_LOCALID")] = None,
    isolate_gpu: Annotated[bool, typer.Option(help="also set CUDA_VISIBLE_DEVICES to the rank's GPU")] = False,
) -> None:
    """Bind this rank's CPUs, memory and env per the plan, then exec CMD. Use under torchrun/mpirun/srun."""
    execwrap.run(_plan(plan_file, topo_file, snapshot, variant), cmd, local_rank, isolate_gpu)


@app.command()
def observe(
    cmd: Annotated[list[str] | None, typer.Argument(help="command to run, after --")] = None,
    duration: Annotated[float | None, typer.Option(help="instead of a command, watch all QPs here for N s")] = None,
    interval: Annotated[float, typer.Option(help="sampling interval in seconds")] = 0.5,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="write the observation as JSON")] = None,
) -> None:
    """Measure live QPs per NIC (`rdma resource show qp`) for a command, or for the whole host over --duration."""
    if bool(cmd) == (duration is not None):
        raise typer.BadParameter("give either a command after -- or --duration")
    obs = obsmod.observe_command(cmd, interval) if cmd else obsmod.observe_host(duration, interval)
    if output:
        output.write_text(json.dumps(asdict(obs), indent=2) + "\n")
    report.show_observation(obs, console)
    if obs.exit_code:
        raise typer.Exit(obs.exit_code)


@app.command()
def calibrate(
    hca: Annotated[str, typer.Option(help="client-side RDMA device, e.g. mlx5_ib0")],
    port: Annotated[int, typer.Option(help="device port")] = 1,
    server: Annotated[str, typer.Option(help="host for the perftest server (over ssh); default: this host")] = "",
    server_hca: Annotated[str | None, typer.Option(help="server-side device (default: --hca)")] = None,
    connect_to: Annotated[
        str | None, typer.Option(help="address the client dials (default: --server or 127.0.0.1)")
    ] = None,
    qps: Annotated[str, typer.Option(help="comma-separated QP counts to sweep")] = ",".join(map(str, cal.DEFAULT_QPS)),
    size: Annotated[int, typer.Option(help="message size in bytes")] = 64,
    duration: Annotated[int, typer.Option(help="seconds per point")] = 5,
    tx_depth: Annotated[int, typer.Option(help="perftest -t per QP")] = 16,
    tool: Annotated[str, typer.Option(help="perftest binary")] = "ib_write_bw",
    gid_index: Annotated[int | None, typer.Option(help="RoCE v2 GID index (Ethernet only)")] = None,
    output: Annotated[Path, typer.Option("-o", "--output", help="profile JSON to write")] = Path("profile.json"),
    name: Annotated[str, typer.Option(help="profile name")] = "",
) -> None:
    """Sweep perftest QP counts with small messages, fit the message-rate knee and write a NIC profile."""
    if not shutil.which(tool):
        raise typer.BadParameter(f"{tool} not found; install perftest")
    topo = discover()
    nic = next((n for n in topo.nics if n.name == hca and n.port == port), None)
    if nic is None:
        console.print(f"[yellow]warning:[/] {hca}:{port} not found locally; profile will not be keyed to a NIC model")
    client = perftest.Endpoint(hca, port, gid_index=gid_index)
    srv = perftest.Endpoint(server_hca or hca, port, host=server, gid_index=gid_index)
    points = cal.sweep(
        srv,
        client,
        connect_to or server or "127.0.0.1",
        [int(q) for q in qps.split(",")],
        tool,
        size,
        duration,
        tx_depth,
        log=console.print,
    )
    profile = cal.to_profile(points, nic, hca, tool, size, name)
    output.write_text(json.dumps(profile, indent=2) + "\n")
    knee = profile["knee_qps"] if profile["knee_found"] else f"not reached (>{profile['knee_qps']})"
    console.print(f"knee: {knee} QPs; profile written to {output}")


@app.command("gdr-matrix")
def gdr_matrix(
    topo_file: TopoFile = None,
    remote: Annotated[str, typer.Option(help="host for the receiving side (over ssh); default: loopback")] = "",
    connect_to: Annotated[str | None, typer.Option(help="address the client dials")] = None,
    size: Annotated[int, typer.Option(help="message size in bytes")] = 1 << 20,
    duration: Annotated[int, typer.Option(help="seconds per pair")] = 3,
    tool: Annotated[str, typer.Option(help="perftest binary built with CUDA")] = "ib_write_bw",
    gid_index: Annotated[int | None, typer.Option(help="RoCE v2 GID index (Ethernet only)")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="write cells as JSON")] = None,
) -> None:
    """GPUDirect RDMA bandwidth from every GPU through every NIC, grouped by path class."""
    if not perftest.supports_cuda(tool):
        raise typer.BadParameter(f"{tool} is missing or was built without --use_cuda support")
    topo = discover(topo_file=topo_file)
    nics = placement_nics(topo)
    cells = gm.measure(
        topo,
        topo.gpus,
        nics,
        remote,
        connect_to or remote or "127.0.0.1",
        tool,
        size,
        duration,
        gid_index,
        log=console.print,
    )
    if output:
        output.write_text(json.dumps([asdict(c) for c in cells], indent=2) + "\n")
    report.show_gdr(cells, gm.by_path_class(cells), console)


@app.command()
def validate(
    launcher: Annotated[str, typer.Option(help='e.g. "mpirun -np 16 -H a:8,b:8 --bind-to none"')],
    tests: Annotated[str, typer.Option(help="comma-separated nccl-tests binaries")] = "alltoall_perf,all_reduce_perf",
    nccl_tests_dir: Annotated[Path | None, typer.Option(help="directory holding the binaries")] = None,
    args: Annotated[str, typer.Option(help="nccl-tests arguments")] = "-b 8 -e 1G -f 4 -g 1",
    variants: Annotated[str, typer.Option(help="none (no binding), best, worst")] = "none,best,worst",
    topo_file: TopoFile = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="write runs as JSON")] = None,
) -> None:
    """Compare nccl-tests bus bandwidth across placements; every rank plans on its own node."""
    exec_args = ["--topo-file", str(topo_file.resolve())] if topo_file else []
    runs = []
    for test in tests.split(","):
        binary = str(nccl_tests_dir / test) if nccl_tests_dir else test
        for variant in variants.split(","):
            runs.append(val.run(launcher, variant, binary, shlex.split(args), exec_args, log=console.print))
    if output:
        output.write_text(json.dumps([asdict(r) for r in runs], indent=2) + "\n")
    report.show_validation(runs, console)


@app.command()
def ui(
    topo_file: TopoFile = None,
    snapshot: Snapshot = None,
    results: Annotated[Path, typer.Option(help="campaign results directory")] = Path("results"),
    profile: Annotated[Path | None, typer.Option(help="extra NIC calibration profile JSON")] = None,
    host: Annotated[str, typer.Option(help="address to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port to listen on (0: any free port)")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="open a browser tab")] = True,
) -> None:
    """Serve a local web page: topology, placement, connection budget and campaign results."""
    uiserver.serve(uiserver.State(_topology(topo_file, snapshot), results, profile), host, port, open_browser)


if __name__ == "__main__":
    app()
