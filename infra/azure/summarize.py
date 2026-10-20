"""Summarize a campaign results directory as Markdown: GDR by path class, QP knee, model vs observed QPs, validation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from railmap.probe import validate

SMALL = 1024  # bytes; message-rate-bound sizes for all-to-all


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(d: Path) -> None:
    print(f"# Campaign {d.name}\n")

    if (d / "gdr.json").exists():
        cells = json.loads((d / "gdr.json").read_text())
        by: dict[str, list[float]] = {}
        for c in cells:
            if c["bw_avg_gbps"] is not None:
                by.setdefault(c["path"], []).append(c["bw_avg_gbps"])
        rows = [[p, f"{sum(v) / len(v):.1f}", f"{min(v):.1f}", f"{max(v):.1f}", len(v)] for p, v in sorted(by.items())]
        print("## GPUDirect RDMA bandwidth by GPU-NIC path class\n")
        print(table(["path", "mean Gb/s", "min", "max", "pairs"], rows) + "\n")

    if (d / "profile.json").exists():
        prof = json.loads((d / "profile.json").read_text())
        knee = prof["knee_qps"] if prof.get("knee_found") else f"not reached (> {prof['knee_qps']})"
        print(f"## QP knee\n\n{prof['source']}\n\nknee: **{knee}** QPs\n")
        print(
            table(
                ["QPs", "Mpps", "relative"],
                [[q, f"{m:.2f}", f"{r:.2f}"] for (q, m, _), (_, r) in zip(prof["raw"], prof["curve"])],
            )
            + "\n"
        )

    rows = []
    for obs_path in sorted(d.glob("obs-q*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
        q = int(re.findall(r"\d+", obs_path.stem)[0])
        obs = json.loads(obs_path.read_text())
        model = json.loads((d / f"model-q{q}.json").read_text())["budget"]["qps_per_nic"]
        observed = max(obs["peak_per_nic"].values(), default=0)
        _, perf = validate.parse((d / f"a2a-q{q}.txt").read_text())
        small = [t for size, t, _ in perf if size <= SMALL]
        err = f"{observed / model - 1:+.0%}" if model else ""
        rows.append([q, model, observed, err, f"{sum(small) / len(small):.1f}" if small else "-"])
    if rows:
        print("## QPs per NIC: model vs observed (2 nodes, alltoall_perf)\n")
        print(table(["QPs/conn", "model", "observed peak", "error", f"mean time <= {SMALL} B (us)"], rows) + "\n")

    if (d / "validate.json").exists():
        runs = json.loads((d / "validate.json").read_text())
        base = {r["test"]: r["avg_busbw"] for r in runs if r["variant"] == "none"}
        vrows = []
        for r in runs:
            ref = base.get(r["test"])
            delta = f"{r['avg_busbw'] / ref - 1:+.1%}" if r["avg_busbw"] and ref and r["variant"] != "none" else ""
            vrows.append([r["test"], r["variant"], r["avg_busbw"], delta, r["exit_code"]])
        print("## Placement validation (nccl-tests average bus bandwidth, GB/s)\n")
        print(table(["test", "variant", "avg busbw", "vs none", "exit"], vrows))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
