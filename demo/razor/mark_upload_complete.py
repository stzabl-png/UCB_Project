#!/usr/bin/env python3
"""
Mark a session input/ upload finished (Razor → Titan daemon trigger).

Run on Razor after rsync, or on Titan:

  python demo/razor/mark_upload_complete.py \\
    --session-dir demo/phase2/sessions/20260602_192346_chips

On Titan the same path under Affordance2Grasp:

  python demo/razor/mark_upload_complete.py \\
    --session-dir demo/sessions/20260602_192346_chips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from demo.pipeline.session_markers import write_upload_complete  # noqa: E402
from demo.pipeline.env import repo_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write input/.upload_complete for Titan daemon")
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--source", type=str, default="razor")
    args = ap.parse_args(argv)

    p = args.session_dir.expanduser()
    if not p.is_absolute():
        p = (repo_root() / p).resolve()
    else:
        p = p.resolve()

    if not (p / "input").is_dir():
        print(f"No input/ under {p}", file=sys.stderr)
        return 1

    out = write_upload_complete(p, source=args.source)
    print(f"Marked upload complete: {out}")
    print("Titan daemon will pick up this session when running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
