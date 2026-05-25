# baseline_3 + cuRobo plan_grasp 长跑污染 bug

**Status**: 2 类不同 bug,各自找到 trigger,fix 部分清楚。

---

## 环境

- **OS / Sim**: Ubuntu, NVIDIA RTX 5090 (32GB), IsaacSim 5.1, **cuRobo 0.8 dev** (`MotionPlanner.plan_grasp`)
- **conda env**: `env_isaaclab` (IsaacSim + 我们的 baseline_3 collector)
- **GPU pid in use**: dp3 inference server ~1.7GB (idle)
- **cuRobo runs out-of-process**: 每次 plan 通过 `subprocess.run([curobo_plan.py, '--grasp', in.pkl, out.pkl])`

## 整体 pipeline

```
sim/run_grasp_sim_baseline3.py (env_isaaclab)
  ├─ 1 IsaacSim 进程 = 1 物体的全 50 ep
  ├─ 每 ep:
  │    1. franka.set_joint_positions(HOME_JOINTS) + pin obj
  │    2. subprocess.run(curobo_plan.py --grasp ...) 子进程 cuRobo MotionPlanner.plan_grasp
  │    3. 收回 approach/grasp/lift 三段 qpos 轨迹
  │    4. 逐 waypoint 驱动: set_joint_positions(qpos[k]) + world.step
  │       (approach + grasp 段 pin obj,lift 段 apply_action)
  │    5. close_gripper + 80 sim steps + read obj_z (success = dz > 3cm)
```

## 现象

**长跑后 cuRobo plan_grasp subprocess 大量失败**(80%+ 失败率,vs 单跑 100% 成功率)。

### Cracker (ycb_dex_02) 51 ep — 典型 type A
- 单 ep 隔离 (`--start 13 --limit 1`): plan SUCCESS, sim 也 OK
- 在 50-ep 连跑里: **ep 1-14 plan OK, ep 15+ ALL plan_failed with `subprocess exception`**, 永久不恢复
- 子进程 exception traceback 截断到 `goalset_result = self.plan_pose(grasp_poses, current_state)` 但具体 exception type 看不到 (我们的 truncation 设 600 字符)

### Sugar (ycb_dex_03) 49 ep — 典型 type B
- 单 ep 隔离 ep 2 (`--start 1 --limit 1`): **plan SUCCESS 但 sim 执行炸**
- 在 50-ep 连跑里: ep 1 OK, ep 2 sim 炸,ep 3+ ALL plan_failed

---

## 两类 bug 分析

### Bug A: Video capture 累积污染(~ep 14 触发 cracker)

**触发**: `--video <dir>` 开启每 N sim 步 `capture_viewport_to_file()` 到 PNG。`~14 ep × 80 sim steps = ~1120 capture` 后,Replicator/Hydra 内部 state 退化 → 后续 cuRobo subprocess CUDA init/run 失败。

**证据**:
| 测试 | plan_ok | plan_fail | 模式 |
|---|---|---|---|
| WITH video (cracker 0-50) | 14/51 | **35/51** (subprocess exception 从 ep 15 起) | 永久 |
| **NO video** (cracker 0-19) | **19/20** | 1/20 (intrinsic Goalset None on ep 16) | 自愈 |

**Fix**: 不开 `--video` 采集,事后用 `sim/gt_replay_ikpd_v2.py --traj saved.hdf5 --video <dir>` 单独 replay 录视频。

### Bug B: PhysX collision blowup(sugar ep 2 触发,**no-video 也炸**)

**触发**: 单个 ep 的 grasp pose 让 panda_hand body 在 grasp 段 (-12cm 沿 tool z 插入物体) 与物体 mesh 深度重叠 → PhysX kinematic-vs-dynamic 物理求解器爆炸,把 Franka leftfinger transform 设成天文数字。

**RTX 报警(关键证据)** (sugar ep 2 isolated, no-video):
```
[Warning] [rtx.scenedb.plugin] Instance 15 of geometry 
"/World/Franka/panda_leftfinger/geometry/panda_leftfinger" 
has bounding box dimensions after transformation exceed 1099511627776 (= 1.1e12).
Transform has scale 0.010000, 
world position (-6900511268514305, -2485871204108935, 5944491361959937)
                             ↑ 6.9e15 米! 浮点 overflow / NaN

[Warning] [rtx.scenedb.plugin] Instance 16 of geometry "/World/Rigid/rigid/textured/textured_006" 
                                                       ↑ sugar 物体 也飞了
```

**关键: cuRobo plan 输出本身 100% clean** (检验过):
```
approach_qpos (62, 7), grasp_qpos (22, 7), lift_qpos (42, 7)
  - 全 joints 都在 Franka 限位内
  - 无 NaN
  - 最大相邻 waypoint joint Δ = 3.2° (非常平滑)
```

So cuRobo plan **完全 valid**,但 IsaacSim 执行该 plan 时 PhysX 爆炸把 Franka 抛到 10^15 米。

**污染传播**: ep 2 把 IsaacSim parent 进程的 Franka articulation / PhysX scene state 搞坏 → ep 3+ 任何 cuRobo subprocess 启动时撞坏的 CUDA/PhysX 上下文 → 抛 exception。

**Fix(部分)**: 
- chunked: 每 N ep 重启新 IsaacSim 进程,污染不跨 batch
- 但 batch 内还是会损失 trigger ep + 后续几个 ep
- 真 fix 要么:(a) 检测 panda_hand body vs object mesh 重叠并跳过,(b) 拒绝该 grasp pose,(c) 用 PD 控制代替 kinematic teleport(慢且不一定解决)

---

## 已经试过的 fix

| 方案 | 结果 |
|---|---|
| 改 chunked (10 ep / batch 新 IsaacSim) | ✓ Type A 完全消除,Type B 只限于单 batch 内 |
| 在每集开头加 `apply_action(joint_positions=HOME)` 清 PD target | ❌ 无效 (假设 PD target 残留是错的) |
| 3-way parallel IsaacSim | ❌ 更糟,GPU 竞争让两类 bug 都恶化 |
| chunked 5 ep / batch | 没测过,理论上对 Type B 更友好 |

## 还没试

- 在 `run_grasp_sim_baseline3.py` 加 per-ep sanity check (drive 完检查 `franka.get_joint_positions()` 是否 NaN/极端,`obj.get_obj_pos()` 是否还在桌附近) → 触发就 early-exit chunk
- 测 `--drive pd` 而不是 `--drive kinematic` (我们只用 kinematic via set_joint_positions)
- 用 `world.reset()` 每 ep (重) — 但物体 USD 需要 reload,慢
- 看 cuRobo plan_grasp 是否能传 `disable_collision_links=["panda_hand"]` 防止 hand body 撞物体 mesh

---

## 🎯 对比 partner main `sim/run_grasp_sim.py` — 发现 4 个差异

### 差异 1: 我们 settle 步数比 partner 少 ~4×

```python
# partner main 每个 grasp candidate 之间:
scene["obj"].rigid.set_linear_velocity(np.zeros(3))
scene["obj"].rigid.set_angular_velocity(np.zeros(3))
scene["franka"].set_joint_positions(HOME_JOINTS)
for _ in range(150):                       # ← 150 sim steps settle
    scene["world"].step(render=RENDER_SIM)

# 我们 main loop:
scene["franka"].set_joint_positions(HOME_JOINTS)
for _ in range(40):                        # ← 只 40 步 settle
    obj.rigid.set_world_pose(obj_pos_w, obj_quat_G)
    obj.rigid.set_linear_velocity(np.zeros(3))
    obj.rigid.set_angular_velocity(np.zeros(3))
    scene["world"].step(render=True)
```

→ 我们 ep 之间 settle 不够,上集物理残余可能没退完。

### 差异 2: 我们没显式 open_gripper 在 plan 前 / 没等待 30 步

```python
# partner 在 plan 前:
franka.open_gripper()
for _ in range(30):                        # ← 30 步给 gripper 完全打开
    world.step(render=RENDER_SIM)

# 我们:
# 在 _execute_grasp_curobo 里 open_gripper + 只 20 步 + 同时 _pin()
```

### 差异 3: partner Grasp pose 有 Z 安全 clamp

```python
# partner 在 plan 前:
TCP_OFFSET = 0.105
pos_world = pos_world - approach_dir * TCP_OFFSET   # contact point → panda_hand
MIN_GRASP_Z = TABLE_TOP_Z + 0.02
if pos_world[2] < MIN_GRASP_Z:
    pos_world[2] = MIN_GRASP_Z                       # clamp 防止 hand 撞桌
```

(partner 是 contact-point sampler;我们是 retarget panda_hand 直接,所以 TCP_OFFSET 不一定适用,但 Z clamp 思想可借鉴)

### 差异 4: partner cuRobo 是 **persistent**(IsaacSim 进程内单 MotionGen),我们是 **per-call subprocess** ⚠️

```python
# partner: 全局单例
global _CUROBO_MG
if _CUROBO_MG is None:
    _CUROBO_MG = init_curobo(scene)        # ← 整个 IsaacSim 进程只 init 一次

# 用法
traj = plan_trajectory(_CUROBO_MG, ...)    # ← 直接调,no subprocess
```

vs 我们:
```python
# 每个 ep 起一个新 subprocess
subprocess.run([..., curobo_plan.py, '--grasp', in.pkl, out.pkl])
```

**Why we use subprocess**: cuRobo 0.8 import 跟 IsaacSim 的 Warp 版本冲突,必须 fresh process。
**Cost**: 每 ep init ~4s warmup + 加重 GPU 内存压力(每次 init 都分配 CUDA buffer)。

**Hypothesis**: 我们的 subprocess 反复 init/destroy 可能加剧 GPU/PhysX 状态不稳定。但**这个不易改**(改回 0.7 老 API 才能 in-process)。

### 同样的没区别

- `execute_trajectory` driver: 完全一样,都 `set_joint_positions + world.step`
- `world.reset()` 都只在 `setup_scene` 一次
- cuRobo plan 参数: partner `max_attempts=10 enable_graph=True enable_opt=True` (0.7 API),我们 plan_grasp 默认

---

## 优先级修复(基于对比)

### Quick wins(不改 cuRobo 架构)

1. **settle 步数 40 → 150** (1 行改) — 给上集物理残余更多时间退干净
2. **加 per-ep 显式 `franka.open_gripper()` + 30 步 wait** before plan_grasp (3 行改) — 确保 gripper 真开
3. **加 per-ep sanity check 早退**: drive 完后检查 `franka.get_joint_positions()` 任一 > 10 rad / NaN → 该 chunk 早退;`obj.get_obj_pos()` xy 偏离 obj_pos_w 超 50cm → 也早退
4. **加 Z safety clamp** (即使 retarget 已带 TCP offset,extra safety 无害)

### 架构级(成本高)

5. **改用 cuRobo 0.7 in-process MotionGen** like partner — 解决 subprocess 反复 init 问题;但需要 downgrade cuRobo,可能 break 其它东西

要不要先做 Quick wins 1+2+3 然后重测?

## 关键文件 (我们 branch `gate3-curobo-ik` uncommitted)

```
sim/run_grasp_sim_baseline3.py       Collector 主脚本 (~870 行)
                                     - main loop (line ~775)
                                     - _execute_grasp_curobo() (line ~410)
                                     - solve_plan_grasp() (line ~201)
                                     - load_object() (line ~337) 
                                     - save_episode_b3() (line ~631)
sim/curobo_plan.py                   cuRobo MotionPlanner.plan_grasp 子进程 wrapper
sim/curobo_world.py                  Mesh 抽取 + world_config (table+ground+obj mesh)
sim/curobo_ik.py                     cuRobo IK chain 子进程 (eval 用,collect 用不到)
sim/run_baseline3_v3_chunked.sh      Chunked collection wrapper (10 ep/batch)
```

## 关键 log

```
/tmp/b3_v3_serial_patched.out        Cracker WITH video, ep 14 后污染 (Bug A 证据)
/tmp/debug_no_video.out              Cracker NO video, ep 1-15 全 ok (Bug A 修复证据)
/tmp/sugar_novideo.out               Sugar NO video, ep 2 仍炸 (Bug B 证据)
/tmp/iso_sugar2.out                  Sugar ep 2 单独 isolated 也炸 (Bug B pose-intrinsic 证据)
/tmp/iso_ep14.out                    Cracker ep 14 单独 isolated 成功 (Bug A 上下文累积证据)
/tmp/sugar_ep2_out.pkl               cuRobo 给 sugar ep 2 的 plan output (检过全 clean)
```

## 相关代码引用

### `_execute_grasp_curobo` 的 drive 循环 (`sim/run_grasp_sim_baseline3.py:~460`)

```python
# approach 段
for qpos7 in a_q:
    grip = franka.get_joint_positions()[7:9]
    franka.set_joint_positions(np.concatenate([qpos7, grip]))    # ← kinematic teleport
    for _ in range(2):
        world.step(render=True)
    _pin()                                                        # 物体 pin 回原位

# grasp 段同样
for qpos7 in g_q:
    franka.set_joint_positions(...)  # ← 这里可能让 hand body 重叠物体
    ...
```

### Partner main 用同样 driver(`sim/run_grasp_sim.py:~566 execute_trajectory`)
我们没仔细对比 driver. partner 可能也有同样 bug,但他用 cuRobo 0.7 老 API + `batch_sim_candidates_pool.py` chunk restart 工程化绕过.

---

## 给 partner 的几个 specific 问题

1. **你的 main `sim/run_grasp_sim.py` 是否见过 Franka leftfinger transform 10^15 米这种 RTX warning?**
2. **你怎么过滤"panda_hand body 在 grasp pose 处与物体 mesh 重叠"的 pose?** 我们用 retarget 直接来的 grasp_pose,有些 pose 让 hand body 嵌入物体。
3. **`batch_sim_candidates_pool.py` 里 `chunk_attempts[ci] += 1` restart 的触发条件是什么?** 是 worker 完全 crash 还是 plan_fail 比例?
4. **0.7 MotionGen vs 0.8 MotionPlanner 是否有 known 稳定性区别?** 我们一开始用 0.8 因为 env 默认装的,但你 main 用 0.7.

---

## 复现步骤

```bash
# 复现 Bug A (video 污染)
cd /home/accelerator/UCB_Project
/home/accelerator/miniforge3/envs/env_isaaclab/bin/python sim/run_grasp_sim_baseline3.py \
    --object ycb_dex_02 --headless \
    --out-dir /tmp/repro_bug_a \
    --video /tmp/repro_bug_a_vid \
# 看 /tmp/repro_bug_a_vid/log ep 15+ 全 plan_grasp failed

# 复现 Bug B (sugar ep 2 collision blowup)
/home/accelerator/miniforge3/envs/env_isaaclab/bin/python sim/run_grasp_sim_baseline3.py \
    --object ycb_dex_03 --headless \
    --start 1 --limit 1 \
    --out-dir /tmp/repro_bug_b
# 看 stdout: "rtx.scenedb.plugin Instance 15 of geometry panda_leftfinger ... 10^15"
```
