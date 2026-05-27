#!/usr/bin/env python3
"""Single-object evaluation runner.

Example:
    sim45 evaluation/eval_single.py --obj-id A16013 \
        --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/A16013_grasp.hdf5 \
        --headless
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-object modular evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obj-id", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--policy", choices=("a2g_pdm",), default="a2g_pdm")
    parser.add_argument("--candidate-hdf5", default=None)
    parser.add_argument(
        "--generate-candidate",
        action="store_true",
        help="Generate A2G/PDM candidates before IsaacSim starts; requires --mesh.",
    )
    parser.add_argument("--mesh", default=None, help="Mesh path for --generate-candidate.")
    parser.add_argument(
        "--candidate-dir",
        default=str(PROJ / "output" / "evaluation" / "candidates"),
        help="Where generated candidate HDF5 files are written.",
    )
    parser.add_argument(
        "--candidate-python",
        default=sys.executable,
        help="Python executable used for --generate-candidate.",
    )
    parser.add_argument("--selection", choices=("top", "index", "sample"), default="top")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--z-yaw-deg", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--result-dir",
        default=str(PROJ / "output" / "evaluation" / "single"),
    )
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--save-hdf5", action="store_true", help="Also write robot_gt-compatible HDF5.")
    return parser


def candidate_output_name(obj_id: str, z_yaw_deg: float | None) -> str:
    if z_yaw_deg is None:
        return f"{obj_id}_grasp.hdf5"
    tag = int(round(float(z_yaw_deg))) % 360
    return f"{obj_id}_yaw{tag:03d}_grasp.hdf5"


def maybe_generate_candidate(args: argparse.Namespace) -> str | None:
    if not args.generate_candidate:
        return args.candidate_hdf5
    if not args.mesh:
        raise ValueError("--generate-candidate requires --mesh")

    out_dir = Path(args.candidate_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / candidate_output_name(args.obj_id, args.z_yaw_deg)
    cmd = [
        args.candidate_python,
        str(PROJ / "tools" / "glb_to_pdm_grasp.py"),
        "--mesh",
        str(Path(args.mesh).expanduser().resolve()),
        "--output-dir",
        str(out_dir),
        "--dataset",
        args.dataset or "evaluation",
        "--z-yaw-deg",
        str(float(args.z_yaw_deg)),
        "--no-vis",
    ]
    print("[eval] generating candidate HDF5 before IsaacSim:")
    print("[eval] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJ), check=True)
    if not out_path.is_file():
        raise FileNotFoundError(f"candidate generation finished but output is missing: {out_path}")
    return str(out_path)


def main() -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args()

    candidate_hdf5 = maybe_generate_candidate(args)
    if not candidate_hdf5:
        raise ValueError("provide --candidate-hdf5 or use --generate-candidate --mesh")

    # IsaacSim must be created before importing modules that touch isaacsim APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from evaluation.policies.a2g_pdm import A2GPDMPolicy, A2GPDMPolicyConfig
        from evaluation.results import append_episode_jsonl, build_episode_record, write_episode_json
        from sim.evaluation.curobo_executor import execute_open_loop_grasp, write_robot_gt_hdf5
        from sim.evaluation.scene_builder import build_scene_spec, setup_scene

        episode_id = args.episode_id or f"{args.obj_id}_{args.policy}_{args.seed:06d}"
        scene_spec = build_scene_spec(
            obj_id=args.obj_id,
            episode_id=episode_id,
            dataset=args.dataset,
            object_scale=args.object_scale,
            sim_z_yaw_deg=args.z_yaw_deg,
            seed=args.seed,
            candidate_hdf5=candidate_hdf5,
        )
        scene = setup_scene(scene_spec, render=not args.headless)

        policy = A2GPDMPolicy(
            A2GPDMPolicyConfig(
                candidate_hdf5=candidate_hdf5,
                selection=args.selection,
                candidate_index=args.candidate_index,
                seed=args.seed,
            )
        )
        policy_output = policy.predict(scene)
        if policy_output.kind != "open_loop_grasp" or policy_output.command is None:
            raise RuntimeError(f"unsupported policy output for first runner: {policy_output.kind}")

        execution = execute_open_loop_grasp(scene, policy_output.command)
        record = build_episode_record(
            scene=scene_spec,
            policy_name=policy.name,
            policy_output=policy_output,
            execution=execution,
        )
        json_path = write_episode_json(record, args.result_dir)
        jsonl_path = append_episode_jsonl(record, args.result_dir)
        print(f"[eval] wrote episode JSON: {json_path}")
        print(f"[eval] appended JSONL: {jsonl_path}")

        if args.save_hdf5:
            h5_path = write_robot_gt_hdf5(
                result_dir=args.result_dir,
                scene=scene_spec,
                command=policy_output.command,
                execution=execution,
                policy_name=policy.name,
            )
            print(f"[eval] wrote robot_gt HDF5: {h5_path}")

        print(
            f"[eval] result: {'SUCCESS' if execution.success else 'FAILED'}"
            f" z_delta={execution.z_delta_m}"
            f" failure_stage={execution.failure_stage}"
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

