"""Collect every experiment in experiments.yaml into docs/data/.

    python -m collector                 # all experiments
    python -m collector --only foody    # one experiment
    python -m collector --report        # also print the model pairing table
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .core import write_json
from .experiments import ADAPTERS

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "experiments.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "data")
    ap.add_argument("--only", nargs="*", help="experiment ids to collect (default: all)")
    ap.add_argument("--report", action="store_true", help="print the pairing table for each experiment")
    args = ap.parse_args(argv)

    base = args.config.resolve().parent
    specs = yaml.safe_load(args.config.read_text())["experiments"]
    index_path = args.out / "experiments.json"
    index = {}
    if index_path.exists():
        import json
        index = {e["id"]: e for e in json.loads(index_path.read_text()).get("experiments", [])}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for spec in specs:
        if args.only and spec["id"] not in args.only:
            continue
        adapter = ADAPTERS[spec["adapter"]]
        registry = yaml.safe_load((base / spec["registry"]).read_text()) if spec.get("registry") else {}
        data = adapter.collect(spec, base, registry)

        data["manifest"].update(id=spec["id"], title=spec["title"], summary=spec.get("summary", ""), updated=now)
        out = args.out / spec["id"]
        sizes = {name: write_json(out / f"{name}.json", data[name]) for name in ("manifest", "models", "evals", "results", "samples") if name in data}
        print(f"[{spec['id']}] wrote " + ", ".join(f"{k}.json {v / 1024:.0f} KB" for k, v in sizes.items()))
        index[spec["id"]] = {
            "id": spec["id"], "title": spec["title"], "summary": spec.get("summary", ""),
            "updated": now, "counts": data["manifest"]["counts"],
            "methods": [m["label"] for m in data["manifest"]["methods"]],
        }
        if args.report:
            print(adapter.report(data))

    order = [s["id"] for s in specs]
    write_json(index_path, {"experiments": [index[i] for i in order if i in index]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
