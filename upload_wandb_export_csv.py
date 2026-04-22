#!/usr/bin/env python3
"""
Read a Weights & Biases table CSV export and re-log scalars to a new W&B run.

Typical export columns:
  "Step","run_display_name - env/metric",...

Requires: pip install wandb
Auth: set WANDB_API_KEY or run `wandb login`.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


def _normalize_step_key(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    for name in fieldnames:
        if name.strip().lower() == "step":
            return name
    return None


def _csv_metric_key(column: str) -> str:
    """Map W&B export column title to a metric key for logging."""
    s = column.strip()
    if " - " in s:
        return s.split(" - ", 1)[1].strip()
    return s


def _to_number(raw: str) -> float | int | None:
    raw = raw.strip()
    if raw == "":
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return None


def load_rows(
    csv_path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        step_col = _normalize_step_key(list(reader.fieldnames))
        if step_col is None:
            raise ValueError('CSV must contain a "Step" column')
        metric_cols = [c for c in reader.fieldnames if c != step_col]
        rows: list[dict[str, Any]] = []
        for row in reader:
            if not any((row.get(c) or "").strip() for c in reader.fieldnames):
                continue
            step_raw = (row.get(step_col) or "").strip()
            if step_raw == "":
                continue
            step_val = _to_number(step_raw)
            if step_val is None:
                continue
            if isinstance(step_val, float) and step_val.is_integer():
                step_scalar: int | float = int(step_val)
            else:
                step_scalar = step_val
            payload: dict[str, Any] = {"__step__": step_scalar}
            for col in metric_cols:
                raw = row.get(col, "") or ""
                val = _to_number(raw)
                if val is not None:
                    payload[_csv_metric_key(col)] = val
            if len(payload) > 1:
                rows.append(payload)
        return metric_cols, rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "csv_path",
        nargs="?",
        default="wandb_export_2026-04-22T00_27_54.719+08_00.csv",
        help="Path to W&B-exported CSV (default: %(default)s)",
    )
    p.add_argument("--project", required=True, help="W&B project name")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity (team/user); else WANDB_ENTITY or default")
    p.add_argument("--run-name", default=None, help="New run name (default: derived from CSV filename)")
    p.add_argument("--group", default=None, help="Optional W&B group")
    p.add_argument("--tags", nargs="*", default=[], help="Optional tags for the run")
    p.add_argument("--dry-run", action="store_true", help="Parse CSV and print first rows only; do not upload")
    args = p.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    metric_cols, rows = load_rows(csv_path)
    if not rows:
        print("No data rows parsed.", file=sys.stderr)
        return 1

    run_name = args.run_name or f"csv-import-{csv_path.stem}"

    if args.dry_run:
        print(f"Parsed {len(rows)} steps, metrics: {[ _csv_metric_key(c) for c in metric_cols ]}")
        for r in rows[:3]:
            print(r)
        return 0

    try:
        import wandb
    except ImportError:
        print("Install wandb: pip install wandb", file=sys.stderr)
        return 1

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=run_name,
        group=args.group,
        tags=list(args.tags),
        config={
            "import_source": "wandb_export_csv",
            "csv_path": str(csv_path),
            "num_steps": len(rows),
        },
    )
    try:
        for r in rows:
            step = r.pop("__step__")
            wandb.log(r, step=step)
    finally:
        run.finish()
    print(f"Logged {len(rows)} steps to {args.entity or ''}/{args.project}/{run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
