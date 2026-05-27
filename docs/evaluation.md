# Evaluation

本文档描述当前第一版 evaluation pipeline。它的目标是把评估主流程从 Isaac Sim 细节里拆出来：`evaluation/` 负责 episode、policy adapter、结果格式，`sim/evaluation/` 负责 Isaac Sim 场景、Franka、RigidObject 和 cuRobo 执行。

当前版本是初版，只支持：

- 单物体
- 单次 rollout
- open-loop grasp pose policy
- A2G/PDM candidate HDF5 输入
- Isaac Sim + cuRobo 执行抓取、闭夹爪、lift、判定成功

后续可以在同一接口下扩展到多物体、多初始 pose、并行 worker、GraspNet、Diffusion Policy/DP3 等方法。

## 目录结构

```text
evaluation/
  eval_single.py              # 单物体单次 evaluation CLI
  specs.py                    # 可序列化的 SceneSpec / PolicyOutput / ExecutionResult
  results.py                  # JSON / JSONL 结果写出
  policies/
    base.py                   # policy adapter 抽象接口
    a2g_pdm.py                # 当前 A2G/PDM candidate HDF5 adapter

sim/evaluation/
  scene_builder.py            # Isaac Sim scene setup, object placement, cuRobo mesh extraction
  context.py                  # SimEvaluationContext runtime handles
  curobo_executor.py          # cuRobo open-loop grasp executor
```

设计原则：

- `evaluation/` 尽量不依赖 Isaac Sim，适合写主流程、接口和结果格式。
- `sim/evaluation/` 可以 import Isaac Sim、cuRobo、Franka 和 RigidObject。
- policy 不直接 step simulation。policy 读取 context 后返回动作意图，executor 负责实际仿真执行和成功判定。

## 环境

Isaac Sim 需要用安装目录里的 `python.sh` 启动。不要依赖本地 alias，例如 `sim45`，因为不同机器不一定配置了它。

```bash
export PROJ=/home/vision/Project/Affordance2Grasp
export ISAAC_SIM_PATH=/home/vision/isaacsim
cd "$PROJ"
```

如果在 conda 环境中启动 Isaac Sim 遇到 numpy/scipy 路径混用问题，可以用更干净的环境变量启动：

```bash
env -u PYTHONPATH -u PYTHONHOME $ISAAC_SIM_PATH/python.sh evaluation/eval_single.py --help
```

## 最小用法：已有 Candidate HDF5

如果已经有 A2G/PDM candidate HDF5：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id A16013 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/A16013_grasp.hdf5 \
  --headless \
  --result-dir output/evaluation/single \
  --save-hdf5
```

这条命令会：

1. 启动 Isaac Sim。
2. 读取 candidate HDF5。
3. 默认选择 score 最高的 candidate。可用 `--selection index --candidate-index N` 或 `--selection sample` 改变选择方式。
4. 查找 `A16013.usd`：
   - `output/obj_usd/{oakink,ycb,arctic,dexycb,egocentric,ho3d_v3}/A16013.usd`
   - `sim/assets/A16013.usd`
5. 按 `sim/run_grasp_sim.py` 的默认场景放置 Franka、桌子和物体。
6. 从 Isaac stage 提取物体 mesh，加入 cuRobo collision world。
7. 将 candidate 的 object-mesh grasp pose 转成 world pose，并适配到 Franka `panda_hand` frame。
8. cuRobo 规划并执行：
   - pre-grasp：带 object mesh 避障
   - final approach：清掉 object mesh，只保留桌面/地面，允许夹爪接触物体
   - lift：闭合夹爪后上提
9. 用 `object_final_z - object_initial_z > 0.03m` 判定 success。

## 可选：先生成 Candidate

对于 real-machine mesh，可以让 runner 先调用现有 A2G/PDM 生成 candidate，再启动 Isaac Sim：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id IMG_4477 \
  --mesh data_hub/real_machine/sam3d_glb/IMG_4477.glb \
  --generate-candidate \
  --candidate-python python \
  --headless
```

注意：

- `--candidate-python python` 通常指当前 shell 里的 conda Python，用来跑 `tools/glb_to_pdm_grasp.py`。
- 仿真阶段仍然需要 Isaac Sim 可加载的 USD。当前会按 obj id 搜索 `output/obj_usd/.../{obj_id}.usd` 或 `sim/assets/{obj_id}.usd`。
- 如果 real-machine 物体还没有 USD，需要先转换 USD，或后续给 runner 增加 `--usd-path` 接口。

## 输出

默认输出目录是 `output/evaluation/single/`，可用 `--result-dir` 修改。

每次 episode 会写：

```text
output/evaluation/single/
  {episode_id}.json
  episodes.jsonl
  {episode_id}_robot_gt.hdf5    # 仅在传入 --save-hdf5 时写出
```

默认 `episode_id`：

```text
{obj_id}_{policy}_{seed:06d}
```

例如：

```text
A16013_a2g_pdm_000000
```

JSON 包含：

- `scene`：USD 路径、object pose、z-yaw、table/robot pose、object scale
- `policy_output`：选中的 candidate、score、gripper width、frame、mesh prerotation
- `execution`：规划阶段状态、executed panda hand pose、gripper tips、初始/最终物体位置
- `success`
- `failure_stage`
- `z_delta_m`

常见 `failure_stage`：

- `target_transform`：policy 输出无法转换成当前 executor 支持的 target
- `curobo_init`：cuRobo 初始化失败
- `pregrasp_plan`：pre-grasp 和 direct plan 都失败
- `final_plan`：最后接近阶段规划失败
- `lift_plan`：lift 规划失败
- `lift_result`：执行完成但物体没有被 lift 超过 3cm

## 当前 Policy 接口

当前支持的 policy output 是 `OpenLoopGraspCommand`：

```python
OpenLoopGraspCommand:
  position              # grasp position, 当前 frame=object_mesh
  rotation              # 3x3 rotation, 当前 frame=object_mesh
  gripper_width
  frame                 # 当前支持 "object_mesh"
  ee_frame_convention   # 当前为 "a2g_grasp_frame"
  name
  score
  mesh_prerotation_euler
  metadata
```

当前 executor 只支持：

```text
PolicyOutput.kind == "open_loop_grasp"
command.frame == "object_mesh"
```

## 后续扩展接口

### GraspNet / 其他 grasp pose 方法

建议新增：

```text
evaluation/policies/graspnet.py
```

adapter 负责把方法输出统一成 `OpenLoopGraspCommand`。如果方法输出在 camera/world/object frame，需要在 adapter 中显式转换到 `object_mesh` 或在 executor 中新增 frame 支持。

### Diffusion Policy / DP3

DP3 这类 closed-loop policy 不适合只返回一个 grasp pose。建议扩展：

```python
PolicyOutput.kind = "closed_loop_actions"
```

然后新增 executor：

```text
sim/evaluation/closed_loop_executor.py
```

它应负责：

- 每个 sim step 读取 observation
- 构造 policy 输入，例如 point cloud、EE pose、gripper state
- 调 policy server 或本地 policy
- 执行动作序列
- 统一 success 判定和结果写出

现有 `sim/eval_dp3_policy.py` 可作为 DP3 closed-loop adapter 的参考。

### 多物体 / 多 rollout / 并行

后续不应改 executor 主逻辑，而是生成多个 `SceneSpec`：

```text
SceneSpec(obj=A, yaw=0, seed=0)
SceneSpec(obj=A, yaw=90, seed=1)
SceneSpec(obj=B, yaw=0, seed=2)
...
```

然后由 batch runner 顺序或并行调度。需要吞吐时，可以借鉴 `sim/run_grasp_sim_pool.py` 的长驻 Isaac worker 和 object swap/reset 逻辑。

## 已知日志噪音

第一次 setup scene 时可能看到一些类似：

```text
/World/Table does not exist
/World/Rigid/rigid_0 does not exist
```

这是清理旧 prim 时删除了不存在的 prim，通常不影响结果。后续可以用 `safe_delete_prim()` 降低日志噪音。

Isaac Sim 还可能输出 GPU dynamics/CCD、Franka mimic joint、mesh normal、TGS velocity iterations 等 warning。只要 episode JSON 中 `success=true` 且 `z_delta_m > 0.03`，当前 evaluation 结果可视为有效。

