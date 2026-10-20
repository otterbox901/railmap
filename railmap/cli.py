"""railmap command line."""

from __future__ import annotations

import json
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
from railmap.topology.discover import discover, placement_nics
from railmap.topology.model import Topology

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


if __name__ == "__main__":
    app()
