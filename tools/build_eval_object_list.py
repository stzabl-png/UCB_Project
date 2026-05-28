#!/usr/bin/env python3
"""Build evaluation object CSV from pool robot_gt success counts (round >= min_round)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
if str(PROJ / "tools") not in sys.path:
    sys.path.insert(0, str(PROJ / "tools"))

from grasp_pool_common import scan_success_round_ge3  # noqa: E402

DEFAULT_OUTDIR = PROJ / "output" / "grasp_collect_no_rot"
DEFAULT_OUTPUT = PROJ / "evaluation" / "configs" / "eval_objects_merged_success_ge30.csv"


def build_list(
    *,
    outdir: Path,
    min_success: int,
    min_round: int,
) -> list[dict[str, str]]:
    counts = scan_success_round_ge3(str(outdir), min_round=int(min_round))
    rows = [
        {
            "obj_id": obj_id,
            "enabled": "1",
            "success_count": str(int(n)),
            "notes": f"round_ge{min_round}_success_ge{min_success}",
        }
        for obj_id, n in counts.items()
        if int(n) >= int(min_success)
    ]
    rows.sort(key=lambda r: (-int(r["success_count"]), r["obj_id"]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build eval object list from round>=N success counts")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--min-success", type=int, default=20)
    p.add_argument("--min-round", type=int, default=3, help="Count successes only from round_R with R >= this")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    rows = build_list(
        outdir=args.outdir.expanduser().resolve(),
        min_success=args.min_success,
        min_round=args.min_round,
    )
    if not rows:
        raise SystemExit(
            f"No objects with success>={args.min_success} from round>={args.min_round} under {args.outdir}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["obj_id", "enabled", "success_count", "notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} objects -> {args.output}")
    print(f"  criterion: round>={args.min_round} success_count>={args.min_success}")
    print(f"  success_count range: {rows[-1]['success_count']} .. {rows[0]['success_count']}")


if __name__ == "__main__":
    main()
