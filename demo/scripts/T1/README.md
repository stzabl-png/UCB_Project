# T1 — Validate input

Checks Razor `input/` against Phase 2 schema **1.1** before running segmentation / SAM3D / FP.

## Usage

From repo root (`Affordance2Grasp`):

```bash
# Session root (contains input/)
python demo/scripts/T1/validate_input.py \
  --session-dir demo/sessions/20260602_192346_chips

# Or point directly at input/
python demo/scripts/T1/validate_input.py \
  --input-dir demo/sessions/20260602_192346_chips/input

# Machine-readable report
python demo/scripts/T1/validate_input.py --session-dir ... --json

# Fail on warnings too (e.g. CI)
python demo/scripts/T1/validate_input.py --session-dir ... --strict-warnings

# Write output/status.json on validation failure (for pipeline abort)
python demo/scripts/T1/validate_input.py --session-dir ... --write-status
```

**Exit codes:** `0` = pass, `1` = validation failed (or warnings with `--strict-warnings`), `2` = bad paths/args.

## Requires

- `numpy`, `opencv-python` (`cv2`)
- Optional: `PIL` for RGB-on-disk layout hint

## See also

- [demo/README.md](../../README.md) — full input schema
- [_session_io.py](../_session_io.py) — shared `--session-dir` / `--input-dir` paths
