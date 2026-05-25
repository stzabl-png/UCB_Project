#!/usr/bin/env python3
"""cuRobo GPU IK solver for gt_replay — a stronger replacement for Lula's offline IK.

Why: Lula's 18-seed local search misses solutions that exist. cuRobo runs 1024 seeds in
parallel on the GPU, is collision-aware (self_collision), and returns many candidate IK
branches per frame so a chain can be selected.

Chain selection objective (Baseline-1's goal is that the EE matches the human trajectory):
  • HARD GATE  — continuity: a per-frame joint step above CONT_THRESH is unexecutable by
                 the sim PD (the replay physics breaks), so such transitions are forbidden.
                 This is a feasibility constraint, not a competing objective: a discontinuous
                 chain makes the *executed* final-EE / mean-tracking garbage anyway.
  • OBJECTIVE  — within continuity-feasible chains, minimise IK pose error, with the FINAL
                 frame weighted heavily (末端 EE vs 人手), then all frames (mean tracking).
  • fallback   — if no continuity-feasible chain exists (an unavoidable wrist flip), fall
                 back to the minimax-continuity chain (least-bad, still reported).

Interface:
    cik = CuroboIK(num_seeds=1024, pos_tol=0.015, ori_tol=0.15)
    qpos_list, ok_list = cik.solve_chain(targets, robot_pos, robot_ori_deg)
        targets       : list of (pos_W (3,), quat_wxyz (4,))  — Franka-convention quats,
                        sim-world frame (the same poses Lula's IK is fed)
        robot_pos     : Franka base xyz in sim world
        robot_ori_deg : Franka base [roll,pitch,yaw] deg (only yaw used)
        → qpos_list[i] : np.array(7,) arm joints, or None if frame i is unreachable
        → ok_list[i]   : bool
"""
import numpy as np
from scipy.spatial.transform import Rotation

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
CONT_THRESH = 2.5      # rad — max per-frame joint step the sim PD can still execute
W_FINAL     = 100.0    # weight on the final frame's IK error (末端 EE vs 人手 = top priority)
W_CONT      = 1e-3     # tiny joint-step tie-breaker (continuity = lowest priority, see header)


class CuroboIK:
    def __init__(self, num_seeds=1024, pos_tol=0.015, ori_tol=0.15,
                 return_seeds=1024, self_collision=True):
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, Pose
        self._torch, self._Pose, self._GoalToolPose = torch, Pose, GoalToolPose
        self._return_seeds = int(return_seeds)
        cfg = InverseKinematicsCfg.create(
            robot="franka.yml", self_collision_check=bool(self_collision),
            num_seeds=int(num_seeds), seed_solver_num_seeds=int(num_seeds),
            position_tolerance=float(pos_tol), orientation_tolerance=float(ori_tol),
            success_requires_convergence=True,
            max_batch_size=1, use_cuda_graph=False,
        )
        self._ik = InverseKinematics(cfg)
        self._tool = self._ik.tool_frames[0]
        self._arm_idx = None
        self.cfg_str = (f"cuRobo IK: tool={self._tool} num_seeds={num_seeds} "
                        f"return_seeds={return_seeds} self_collision={self_collision} "
                        f"tol=(pos {pos_tol*1000:.0f}mm, ori {ori_tol})")

    def _solve_one(self, pos_b, quat_b_wxyz):
        """Solve one base-frame pose. Returns (qpos (N,7), pos_err (N,)) for successful
           candidates (cuRobo ranks them best-first); pos_err is the IK residual in metres."""
        torch = self._torch
        goal = self._Pose(
            position=torch.tensor([pos_b], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([quat_b_wxyz], device="cuda", dtype=torch.float32))
        res = self._ik.solve_pose(
            self._GoalToolPose.from_poses({self._tool: goal}, num_goalset=1),
            return_seeds=self._return_seeds)
        if self._arm_idx is None:
            jn = getattr(res.js_solution, "joint_names", None)
            self._arm_idx = ([list(jn).index(j) for j in ARM_JOINTS]
                             if jn is not None else list(range(7)))
        pos = res.js_solution.position.detach().cpu().numpy()[0]               # (K, n_joints)
        succ = np.atleast_1d(res.success.detach().cpu().numpy()[0]).astype(bool)
        perr = np.atleast_1d(res.position_error.detach().cpu().numpy()[0]).astype(np.float64)
        arm = pos[:, self._arm_idx]                                            # (K, 7)
        return arm[succ], perr[succ]

    def solve_chain(self, targets, robot_pos, robot_ori_deg, verbose=True, start_qpos=None):
        """Solve an IK chain.

        start_qpos: optional (7,) ndarray — the joint config the chain should be
            continuous WITH at its left edge. Used to stitch chunked DP3 rollouts:
            pass the LAST executed qpos of the previous chunk so the DP picks IK
            branches for frame 0 that are near it (avoids elbow-flip jumps across
            chunks). Treated internally as a virtual frame -1 with one candidate
            (zero error), so frame 0 candidates are continuity-gated against it.
        """
        robot_pos = np.asarray(robot_pos, dtype=np.float64)
        yaw = robot_ori_deg[2] if hasattr(robot_ori_deg, "__len__") else robot_ori_deg
        Rz_inv = Rotation.from_euler("z", yaw, degrees=True).inv()
        cands = []
        for pos_w, quat_w in targets:
            pw = np.asarray(pos_w, dtype=np.float64)
            qw = np.asarray(quat_w, dtype=np.float64)
            pos_b = Rz_inv.apply(pw - robot_pos)
            Rw = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])              # wxyz → xyzw
            qb = (Rz_inv * Rw).as_quat()                                       # xyzw
            cands.append(self._solve_one(pos_b, [qb[3], qb[0], qb[1], qb[2]]))
        qpos_list, ok_list, info = self._select_chain(cands, start_qpos=start_qpos)
        if verbose:
            ms = _chain_max_step(qpos_list)
            seed_tag = f" seed={'on' if start_qpos is not None else 'off'}"
            print(f"   {self.cfg_str}{seed_tag}")
            print(f"   solved {sum(ok_list)}/{len(ok_list)} frames   "
                  f"IK pos-err: mean {info['mean_err']*1000:.1f}mm  final {info['final_err']*1000:.1f}mm   "
                  f"chain max joint-step = {np.rad2deg(ms):.0f}°   ({info['mode']})")
        return qpos_list, ok_list

    @staticmethod
    def _select_chain(cands, start_qpos=None):
        """Accuracy-first DP under a hard continuity gate (see module header).

        If start_qpos is provided, a virtual seed frame (1 candidate, zero error) is
        prepended so frame 0 of the real chain is continuity-checked against it.
        The returned chain still has length n (the seed itself is not returned).
        """
        n = len(cands)
        Q = [c[0] for c in cands]
        E = [c[1].astype(np.float64) for c in cands]
        if any(len(q) == 0 for q in Q):
            chain, ok = CuroboIK._select_chain_minimax(Q)
            return chain, ok, dict(mode="minimax/holey", mean_err=float("nan"), final_err=float("nan"))

        # Prepend seed as a virtual frame-(-1) with one candidate; DP then naturally
        # continuity-gates real frame 0 against it.
        if start_qpos is not None:
            seed_arr = np.asarray(start_qpos, dtype=np.float64).reshape(1, -1)
            Q = [seed_arr] + Q
            E = [np.zeros(1, dtype=np.float64)] + E
            n_eff = n + 1
            first_real = 1
        else:
            n_eff = n
            first_real = 0

        # forward DP: cost[i] = min total cost of a continuity-feasible chain ending at cand i
        cost = E[0].copy()
        bp = []
        for t in range(1, n_eff):
            step = np.abs(Q[t][:, None, :] - Q[t - 1][None, :, :]).max(axis=2)   # (Kc,Kp) rad
            cc = cost[None, :] + W_CONT * step
            cc[step > CONT_THRESH] = np.inf
            j = np.argmin(cc, axis=1)
            base = cc[np.arange(len(Q[t])), j]
            is_final = (t == n_eff - 1)
            cost = base + (W_FINAL if is_final else 1.0) * E[t]
            bp.append(j)
        i = int(np.argmin(cost))
        if not np.isfinite(cost[i]):                                  # no feasible chain
            # fallback: drop seed gate, run minimax on real frames only
            chain, ok = CuroboIK._select_chain_minimax([c[0] for c in cands])
            return chain, ok, dict(mode="minimax/no-feasible-chain",
                                   mean_err=float("nan"), final_err=float("nan"))
        idx = [i]
        for t in range(n_eff - 1, 0, -1):
            i = int(bp[t - 1][i]); idx.append(i)
        idx.reverse()
        chain = [Q[t][idx[t]] for t in range(first_real, n_eff)]
        errs = [float(E[t][idx[t]]) for t in range(first_real, n_eff)]
        return chain, [True] * n, dict(mode="accuracy-DP",
                                       mean_err=float(np.mean(errs)), final_err=errs[-1])

    @staticmethod
    def _select_chain_minimax(Q):
        """Fallback: minimise the chain's largest joint jump (bottleneck DP). Used when no
           continuity-feasible chain exists, or when some frame has no IK candidate."""
        n = len(Q)
        if any(len(q) == 0 for q in Q):
            return CuroboIK._select_chain_greedy(Q)
        cost = np.zeros(len(Q[0]))
        bp = []
        for t in range(1, n):
            trans = np.abs(Q[t][:, None, :] - Q[t - 1][None, :, :]).max(axis=2)
            cc = np.maximum(cost[None, :], trans)
            j = np.argmin(cc, axis=1)
            cost = cc[np.arange(len(Q[t])), j]
            bp.append(j)
        i = int(np.argmin(cost))
        idx = [i]
        for t in range(n - 1, 0, -1):
            i = int(bp[t - 1][i]); idx.append(i)
        idx.reverse()
        return [Q[t][idx[t]] for t in range(n)], [True] * n

    @staticmethod
    def _select_chain_greedy(Q):
        """Greedy multi-start fallback — used only when some frame has no IK candidate."""
        n = len(Q)
        nonempty = [t for t in range(n) if len(Q[t]) > 0]
        if not nonempty:
            return [None] * n, [False] * n
        best_chain, best_score = None, None
        for start in Q[nonempty[0]]:
            chain = [None] * n
            chain[nonempty[0]] = start
            prev, mx = start, 0.0
            for t in range(nonempty[0] + 1, n):
                if len(Q[t]) == 0:
                    continue
                d = np.abs(Q[t] - prev).max(axis=1)
                k = int(np.argmin(d))
                chain[t] = Q[t][k]; prev = Q[t][k]; mx = max(mx, float(d[k]))
            if best_score is None or mx < best_score:
                best_chain, best_score = chain, mx
        return best_chain, [c is not None for c in best_chain]


def _chain_max_step(qpos_list):
    if any(q is None for q in qpos_list):
        return float("inf")
    return max((np.abs(qpos_list[k] - qpos_list[k - 1]).max()
                for k in range(1, len(qpos_list))), default=0.0)


if __name__ == "__main__":
    import sys

    # ── subprocess mode: `curobo_ik.py --solve IN.npz OUT.npz` ───────────────
    # gt_replay calls this out-of-process — cuRobo 0.8's collision module needs a
    # newer Warp than IsaacSim bundles, so a fresh process gets the correct Warp.
    if len(sys.argv) >= 4 and sys.argv[1] == "--solve":
        in_npz, out_npz = sys.argv[2], sys.argv[3]
        d = np.load(in_npz)
        targets = list(zip(d["pos"], d["quat"]))            # Franka-convention quats
        cik = CuroboIK(num_seeds=int(d["num_seeds"]),
                       pos_tol=float(d["pos_tol"]), ori_tol=float(d["ori_tol"]))
        # Optional warm-start: previous chunk's last qpos. Pass via `start_qpos` key.
        start_qpos = d["start_qpos"] if "start_qpos" in d.files else None
        if start_qpos is not None and start_qpos.size != 7:
            start_qpos = None                                 # ignore malformed seed
        qpos_list, ok_list = cik.solve_chain(targets, d["robot_pos"], d["robot_ori"],
                                             start_qpos=start_qpos)
        out = np.full((len(qpos_list), 7), np.nan, dtype=np.float64)
        for i, q in enumerate(qpos_list):
            if q is not None:
                out[i] = q
        np.savez(out_npz, qpos=out, ok=np.array(ok_list, dtype=bool))
        print(f"[curobo_ik --solve] {sum(ok_list)}/{len(ok_list)} frames → {out_npz}")
        sys.exit(0)

    # ── standalone self-test: run on the 6 Gate-3 FAIL trajectories ──────────
    import h5py
    SIM_ORIGIN_W = np.array([0.0, 0.30, 0.80])
    _SWAP = Rotation.from_euler("z", -90, degrees=True)

    def axis_swap(q):                            # raw retarget quat → Franka convention
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]]) * _SWAP
        o = r.as_quat()
        return np.array([o[3], o[0], o[1], o[2]])

    def auto_robot_pose(s0, sox=(0.0, 0.30), reach=0.35, base_z=0.80):
        out = np.array([s0[0], s0[1]]); nn = np.linalg.norm(out)
        out = out / nn if nn > 1e-3 else np.array([1.0, 0.0])
        return (np.array([s0[0] + sox[0] + reach * out[0], s0[1] + sox[1] + reach * out[1], base_z]),
                [0.0, 0.0, float(np.degrees(np.arctan2(-out[1], -out[0])))])

    OBJS = [("scissors", "ycb17"), ("sugar_box", "ycb03"), ("mustard_bottle", "ycb05"),
            ("potted_meat_can", "ycb09"), ("banana", "ycb10"), ("extra_large_clamp", "ycb20")]
    cik = CuroboIK(num_seeds=1024, pos_tol=0.015, ori_tol=0.15)
    for name, hf in OBJS:
        h = h5py.File(f"/tmp/gate3_sweep/{hf}.hdf5", "r")
        state = h["state"][:]; actions = h["action"][:]; h.close()
        rp, ro = auto_robot_pose(state[0, :3])
        rows = [state[0]] + [actions[t] for t in range(actions.shape[0])]
        targets = [(r[:3] + SIM_ORIGIN_W, axis_swap(r[3:7])) for r in rows]
        print(f"\n=== {name} ===")
        cik.solve_chain(targets, rp, ro)
