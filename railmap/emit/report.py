"""Human-readable tables for topology, plans and connection budgets."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from railmap.doctor import Check
from railmap.placement.assign import Plan
from railmap.placement.budget import Budget
from railmap.probe.gdr_matrix import Cell
from railmap.probe.observe import Observation
from railmap.probe.validate import Run
from railmap.topology.model import PathClass, Topology, format_cpulist

PATH_STYLE = {
    PathClass.PIX: "green",
    PathClass.PXB: "green",
    PathClass.PHB: "yellow",
    PathClass.NODE: "yellow",
    PathClass.SYS: "red",
}


def _messages(console: Console, warnings: list[str], notes: list[str]) -> None:
    for w in warnings:
        console.print(f"[bold red]warning:[/] {w}")
    for n in notes:
        console.print(f"[dim]note: {n}[/]")


def show_topology(topo: Topology, console: Console) -> None:
    console.print(f"[bold]source:[/] {topo.source}")
    if not topo.gpus:
        console.print("no NVIDIA GPUs found")
    if not topo.nics:
        console.print("no RDMA NICs found (/sys/class/infiniband is empty)")

    if topo.gpus and topo.nics:
        t = Table(title="GPU ↔ NIC path class")
        t.add_column("GPU")
        t.add_column("NUMA")
        for j, _ in enumerate(topo.nics):
            t.add_column(f"NIC{j}")
        for g in topo.gpus:
            cells = []
            for n in topo.nics:
                p = topo.path(g, n)
                cells.append(f"[{PATH_STYLE[p]}]{p.name}[/]")
            t.add_row(f"GPU{g.index} {g.bdf}", str(g.numa), *cells)
        console.print(t)
    elif topo.gpus:
        t = Table(title="GPUs")
        for col in ("GPU", "PCI", "NUMA", "name"):
            t.add_column(col)
        for g in topo.gpus:
            t.add_row(str(g.index), g.bdf, str(g.numa), g.name)
        console.print(t)

    if topo.nics:
        t = Table(title="RDMA NICs")
        for col in ("", "NIC", "PCI", "NUMA", "link", "rate", "state", "board", "fw", "netdev"):
            t.add_column(col)
        for j, n in enumerate(topo.nics):
            rate = f"{n.rate_gbps:g} Gb/s" if n.rate_gbps else ""
            t.add_row(
                f"NIC{j}",
                n.label,
                n.bdf or "virtual",
                str(n.numa),
                n.link_layer,
                rate,
                n.state,
                n.board_id,
                n.fw_ver,
                ",".join(n.netdevs),
            )
        console.print(t)

    t = Table(title="NUMA nodes")
    t.add_column("node")
    t.add_column("CPUs")
    for node in topo.numa_nodes:
        t.add_row(str(node.id), format_cpulist(node.cpus))
    console.print(t)
    _messages(console, topo.warnings(), topo.notes)


def show_plan(plan: Plan, console: Console) -> None:
    t = Table(title=f"placement ({plan.variant})")
    for col in ("local rank", "GPU", "NIC", "path", "NUMA", "CPUs"):
        t.add_column(col)
    for r in plan.ranks:
        style = {"PIX": "green", "PXB": "green", "SYS": "red"}.get(r.path or "", "yellow")
        path = f"[{style}]{r.path}[/]" if r.path else "-"
        t.add_row(
            str(r.local_rank),
            f"GPU{r.gpu} {r.gpu_bdf}",
            ", ".join(r.nics) or "-",
            path,
            str(r.numa),
            format_cpulist(r.cpus),
        )
    console.print(t)
    _messages(console, plan.warnings, plan.notes)


def show_budget(b: Budget, console: Console) -> None:
    j = b.job
    console.print(
        f"[bold]job:[/] {j.nodes} nodes x {j.gpus_per_node} GPUs, {j.nics_per_node} NICs/node, "
        f"{j.pattern} over {j.stack}{' (rail-only)' if j.rail_only else ''}"
    )
    style = {"ok": "green", "near": "yellow", "over": "red"}[b.status]
    console.print(f"QPs per NIC (estimate): [bold {style}]{b.qps_per_nic}[/]  = {b.formula}")
    console.print(
        f"profile: {b.profile} ({b.profile_kind}), knee {b.knee_qps} QPs/NIC; "
        f"per-GPU budget {b.per_gpu_budget} QPs; status [bold {style}]{b.status}[/]"
    )
    if b.relative_rate is not None:
        bound = "at most " if b.beyond_curve else ""
        console.print(f"expected small-message rate: {bound}{b.relative_rate:.0%} of peak")
    elif b.profile_kind == "prior":
        console.print("expected small-message rate: unknown below the prior's datum; calibrate this NIC")
    console.print(
        "nodes before the knee: " + (f"{b.max_nodes}" if b.max_nodes is not None else "never (QPs do not grow)")
    )
    if b.recommendations:
        t = Table(title="ways to cut QPs per NIC")
        for col in ("change", "setting", "QPs/NIC", "nodes before knee"):
            t.add_column(col)
        for r in b.recommendations:
            t.add_row(r.change, r.setting, str(r.qps_per_nic), str(r.max_nodes) if r.max_nodes is not None else "any")
        console.print(t)


def show_observation(obs: Observation, console: Console) -> None:
    t = Table(title=f"peak QPs per NIC ({'this command' if obs.mode == 'pid' else 'whole host, minus baseline'})")
    t.add_column("NIC")
    t.add_column("peak QPs")
    for nic, n in sorted(obs.peak_per_nic.items()):
        t.add_row(nic, str(n))
    console.print(t)
    console.print(f"peak total: {obs.peak_total} over {len(obs.timeline)} samples")
    if not obs.peak_per_nic:
        console.print("[yellow]no QPs seen[/] (no RDMA traffic, or `rdma` cannot see other users' QPs without root)")


def show_gdr(cells: list[Cell], summary: dict[str, tuple[float, float, int]], console: Console) -> None:
    gpus = sorted({c.gpu for c in cells})
    nics = list(dict.fromkeys(c.nic for c in cells))
    t = Table(title="GPUDirect RDMA bandwidth, Gb/s")
    t.add_column("GPU")
    for n in nics:
        t.add_column(n)
    for g in gpus:
        row = []
        for n in nics:
            c = next(c for c in cells if c.gpu == g and c.nic == n)
            bw = f"{c.bw_avg_gbps:.0f}" if c.bw_avg_gbps is not None else "fail"
            row.append(f"[{PATH_STYLE[PathClass[c.path]]}]{bw}[/] {c.path}")
        t.add_row(f"GPU{g}", *row)
    console.print(t)
    s = Table(title="by path class")
    for col in ("path", "mean Gb/s", "min Gb/s", "pairs"):
        s.add_column(col)
    for p in sorted(summary, key=lambda p: PathClass[p]):
        mean, low, n = summary[p]
        s.add_row(p, f"{mean:.1f}", f"{low:.1f}", str(n))
    console.print(s)


def show_validation(runs: list[Run], console: Console) -> None:
    t = Table(title="nccl-tests average bus bandwidth, GB/s")
    for col in ("test", "variant", "avg busbw", "vs none", "exit"):
        t.add_column(col)
    base = {r.test: r.avg_busbw for r in runs if r.variant == "none"}
    for r in runs:
        ref = base.get(r.test)
        delta = f"{(r.avg_busbw / ref - 1):+.1%}" if r.avg_busbw and ref and r.variant != "none" else ""
        avg = f"{r.avg_busbw:.2f}" if r.avg_busbw is not None else "-"
        t.add_row(r.test, r.variant, avg, delta, str(r.exit_code))
    console.print(t)
    for r in runs:
        if r.exit_code or r.avg_busbw is None:
            console.print(f"[red]{r.test}/{r.variant} failed[/]: {r.command}\n{r.tail[-800:]}")


DOCTOR_STYLE = {"ok": "green", "warn": "yellow", "fail": "bold red"}


def show_doctor(checks: list[Check], console: Console) -> None:
    t = Table(title="railmap doctor")
    for col in ("check", "status", "detail"):
        t.add_column(col)
    for c in checks:
        detail = c.detail + (f"\n[dim]fix: {c.fix}[/]" if c.fix and c.status != "ok" else "")
        t.add_row(c.name, f"[{DOCTOR_STYLE[c.status]}]{c.status}[/]", detail)
    console.print(t)
