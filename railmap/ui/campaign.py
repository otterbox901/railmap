"""Read a campaign results directory (written by infra/azure/remote-campaign.sh) for the UI."""

from __future__ import annotations

import json
import re
from pathlib import Path

from railmap.probe import validate

SMALL = 1024  # bytes; message-rate-bound sizes, as in infra/azure/summarize.py
MARKERS = ("gdr.json", "profile.json", "validate.json", "plan.json")


def _qps_per_conn(p: Path) -> int:
    return int(re.findall(r"\d+", p.stem)[0])


def list_campaigns(results: Path) -> list[str]:
    """Campaign directory names under `results`, newest first (names end in a UTC stamp)."""
    if not results.is_dir():
        return []
    names = [
        d.name
        for d in results.iterdir()
        if d.is_dir() and (any((d / m).exists() for m in MARKERS) or any(d.glob("obs-q*.json")))
    ]
    return sorted(names, reverse=True)


def load(d: Path) -> dict:
    """Everything the results page draws; a missing file leaves its section empty."""
    out: dict = {"name": d.name, "gdr": None, "profile": None, "inflation": [], "validate": None}
    for key, fname in (("gdr", "gdr.json"), ("profile", "profile.json"), ("validate", "validate.json")):
        if (d / fname).exists():
            out[key] = json.loads((d / fname).read_text())
    for obs_path in sorted(d.glob("obs-q*.json"), key=_qps_per_conn):
        q = _qps_per_conn(obs_path)
        obs = json.loads(obs_path.read_text())
        model_path, a2a_path = d / f"model-q{q}.json", d / f"a2a-q{q}.txt"
        model = json.loads(model_path.read_text())["budget"]["qps_per_nic"] if model_path.exists() else None
        small: list[float] = []
        if a2a_path.exists():
            _, perf = validate.parse(a2a_path.read_text())
            small = [t for size, t, _ in perf if size <= SMALL]
        out["inflation"].append(
            {
                "qps_per_conn": q,
                "model": model,
                "observed": max(obs["peak_per_nic"].values(), default=0),
                "small_time_us": sum(small) / len(small) if small else None,
            }
        )
    return out
