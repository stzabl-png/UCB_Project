# DP3 真机部署 — 完整逻辑链(坐标系为主)

> 状态:逻辑链 v1,已对着 partner 代码逐项验证(2026-06-03)。
> 架构锁定:**Server 推理 + Razor 控制**,session 目录 rsync 传输。
> 目标:DP3(3D Diffusion Policy,点云闭环策略)真机部署,**最大化复用 partner 主方法(Affordance2Grasp)已完成的感知 + retarget + 驱动基建**。

参考代码:
- Server(Titan,`stzabl-png/UCB_Project` branch `titan`):`demo/scripts/T1..T7`、`demo/scripts/T5/mesh_align.py`、`demo/scripts/T6/pdm_session.py`、`tools/{glb_to_pdm_grasp,infer_mesh_v6}.py`、`model/pdm/pose_codec.py`
- Razor(`jiaka1chen/V2AP-demo` branch `main`):`demo/phase2/{retarget,pinch_ik,hand_retarget_geometry,ee_retarget_io}.py`、`demo/phase2/calib/ee_retarget.yaml`、`teleop/{ik_utils,arm_hand_control,robot_descriptions}.py`、`demo/phase1/executor.py`
- DP3 语义(本仓 `gate3-curobo-ik`):`Baseline1/build_gt_replay.py`(训练数据=约定权威源)、`Baseline1/eval/dp3_inference_server.py`、`sim/eval_dp3_titan_protocol.py`

---

## 0. 一句话总览

Razor 拍一帧 RGB-D → rsync 给 Titan → Titan 跑 T1–T5(分割→mesh→尺度→FoundationPose→**base 对齐 mesh**)→ **复用 T6 的采样代码从 base 对齐 mesh 采 4096 点云** → DP3 server 闭环推理(每 chunk 收 Razor 回传的当前 EE proprio,吐 8 帧物体系 EE 位姿)→ Razor 把每帧位姿转回 base、用 **partner 现成 retarget(`base_ee_from_virtual_pinch_closed`)** 求 R_ee → Pink IK 求关节 → **写 partner 现成 300Hz `action_buffer`(自带速度限幅平滑+自碰撞+跟踪安全)** 驱动 Dexmate,直到物体 dz>3cm 或 max_chunks。

---

## 1. 坐标系定义(全程 `p_dst = T_dst_src @ p_src`,列向量)

| 帧 | 记号 | 定义 | 谁产生 |
|---|---|---|---|
| **camera** | `cam` | ZED 左目光学系(+X右 +Y下 +Z前) | Razor 拍照 |
| **robot base** | `base` | Dexmate Vega 根/base;**Z 向上 = 重力**(`table.json`:world Z up,桌面 = z=table_height) | Razor URDF |
| **mesh / 物体自身系** | `mesh` | SAM3D mesh 局部系。T4 均匀缩放(米制),T5 后**已旋转成轴∥base**(见下) | Titan |
| **base 对齐 mesh** | `mesh`(同上) | partner 把 mesh 顶点重标记 `v'=R_b(v−c)+c`,使 mesh 轴∥base 轴,物理位姿不变 → `object_base_aligned.glb`;此时 `T_base_mesh` 的 **R≈I** | `T5/mesh_align.py:94` |
| **DP3 物体系 G** | `G` | DP3 训练用的物体中心系:**原点在物体、轴∥重力/world**。**与 base 对齐 mesh 系等价**(见 §3) | 本部署定义 |
| **pinch / 虚拟夹爪(闭)** | `pinch` | 拇指尖-食指尖**中点**为原点,列=`[finger_open, y_body, approach]` | UCB 约定 |
| **R_ee** | `R_ee` | Razor URDF 上的 Pinocchio 末端帧(= 臂法兰 `R_arm_l8`);**IK 的目标帧** | Razor |
| **DP3 EE** | `ee` | DP3 输入输出的 EE 帧:**朝向 = pinch 朝向**,**原点 = pinch中点 − 0.10·approach** | `build_gt_replay.py:257-258` |

### 1.1 pinch 帧轴约定(DP3 与主方法**完全一致** —— 已验证)

DP3 训练数据(`build_gt_replay.py:240-259`,`mano_to_ee_thumb_index`):
```
ex = normalize(index_tip − thumb_tip)          # 拇指→食指 = finger_open
ez = normalize((thumb+index)/2 − wrist) ⊥ ex   # 腕→pinch = approach(伸向物体)
ey = ez × ex
R  = [ex, ey, ez]
p  = (thumb+index)/2 − EE_OFFSET·ez   # EE_OFFSET = 0.10 m
```
主方法 grasp 旋转(`model/pdm/pose_codec.py:6`):`rotation columns = [finger_dir, lateral_dir, approach_dir]`,approach = 列2。
Razor 从机器人 FK 算 pinch 帧(`hand_retarget_geometry.py:62-78`,`virtual_pinch_frame_in_base`):
```
finger_open = normalize(p_index − p_thumb)     # 拇指→食指
approach    = normalize(p_pinch − p_ee)         # R_ee→pinch
y_body      = normalize(cross(approach, finger_open)); finger_open 重正交
R = [finger_open, y_body, approach];  origin = p_pinch = (p_thumb+p_index)/2
```
**三处逐轴对齐** → DP3 的 EE 朝向 = partner pinch 帧朝向。DP3 EE 与 pinch 只差**沿 approach 的常量 0.10 m**(DP3 EE 在 pinch 后方 0.10m)。

> ⚠️ sim 里那两个不一致的四元数翻转(gate3 `Rz±90°` / titan `diag(1,−1,−1)`)是 **IsaacSim 从 `panda_hand` USD 链接读朝向**的仿真产物,真机不经过 panda_hand,**不传递到真机**。真机走 pinch→R_ee(partner `T_ee_pinch` 标定)即可。

---

## 2. 关键变换(全链)

```
T_base_cam      : Razor extrinsics.json(拍照时 FK)        cam  → base
T_cam_mesh      : FoundationPose 输出                       mesh → cam
T_base_mesh     : = T_base_cam @ T_cam_mesh,R≈I           mesh → base   ← Titan 写 register/T_base_mesh.json
T_ee_pinch_closed : partner 标定(ee_retarget.yaml)        pinch→ R_ee   ← 复用,不重标
```
物体系原点在 base 的位置:`origin_base = T_base_mesh[:3,3]`。

---

## 3. G 系 ↔ base 对齐 mesh 系:为什么只差 translation

- partner 已让 `object_base_aligned.glb` 的 mesh 轴 **∥ base 轴**(`T_base_mesh` 的 R≈I)。
- base 的 Z 轴 = world up = 重力(Vega 直立)。
- DP3 的 G 系定义 = 物体中心、轴∥重力/world。
- 三者轴向一致 → **mesh(base对齐)系 = G 系,二者与 base 只差一个平移 `origin_base`**。
- 因此 base ↔ G 的换算**就是一个平移**(用户理解正确):`p_G = p_base − origin_base`,朝向不变。

> 严格写法用 `inv(T_base_mesh)`(含 R 的 0.02rad 残差);一阶就是减 `origin_base`。**开发统一用 `inv(T_base_mesh)`** 保证与 partner 对齐方式完全一致,不引入额外近似。

---

## 4. Server 端逻辑(Titan)

复用 T1–T5 **一字不改**,得到 `object_base_aligned.glb` + `register/T_base_mesh.json`。新增 `T6_dp3`:

1. **采点云**(复用 T6 采样代码 `tools/infer_mesh_v6.py:288-298` 经 `glb_to_pdm_grasp.prepare_mesh_item`):
   - 对 `object_base_aligned.glb` 做 `trimesh.sample.sample_surface(mesh, 4096, seed)`。
   - **无 center / 无 normalize / 无再缩放 / 无 +90°X**(和主方法 PDM 完全相同的 flag)。
   - 得到 `pts_mesh (4096,3)`,在 base 对齐 mesh 系。
2. **G 系点云 = mesh 帧顶点本身**:`pc_G = pts_mesh`(直接是采样得到的 `object_base_aligned.glb` 顶点,已在 mesh 帧 = G 帧,**和 partner PDM 喂的点云逐点一致**,零变换)。**整个 episode 固定**,只采一次。
   > 定义 **G 帧 ≡ mesh(base 对齐)帧**。base↔G 用 **`inv(T_base_mesh)` / `T_base_mesh`**(§3),因 R≈I 故"≈纯平移"但严格含 ≤1° 残差。EE(§5)经 `inv(T_base_mesh)` 落入**与点云同一个 mesh 帧** → 点云与 EE 严格同帧,相对几何零误差。
3. **state0 = 固定 HOME**:DP3 的初始 EE(物体系)。与 sim 一致,真机起手位姿也从同一 HOME 出发(关节 HOME → FK → §6 算 ee_G)。
4. **起 DP3 server**(复用 `Baseline1/eval/dp3_inference_server.py`,`/predict`):请求 `{point_cloud:[T,4096,3], agent_pos:[T,8]}`,响应 `{action:[8,8]}`。点云这一路固定,proprio 由 Razor 每 chunk 回传。
5. 导出 `output/inference/dp3_session.json`:`{T_base_mesh, origin_base, pc_G(或指针), state0_home, n_obs, server_addr}`。

> DP3 推理本身在 server;点云 server 持有,proprio 来自 Razor。每 chunk 一个 round-trip(8 个 waypoint,不是每 tick),网络延迟可接受。

---

## 5. proprio:从真机算 DP3 的 8D EE 观测(Sharpa,不是 Franka)

每个 chunk,Razor 读当前关节 → Pinocchio 全身 FK →

1. `T_base_pinch = virtual_pinch_frame_in_base(model, data)`（`hand_retarget_geometry.py:62`,复用）。朝向 = DP3 约定,原点 = pinch 中点。
2. **DP3 EE 帧(base)**:朝向 = `T_base_pinch[:3,:3]`;原点 = `p_pinch − 0.10·approach`,`approach = T_base_pinch[:3,2]`。→ `T_base_ee_dp3`。
3. **转物体系 G**:`T_G_ee = inv(T_base_mesh) @ T_base_ee_dp3`。
4. `agent_pos = [ pos_G(3), quat_G_wxyz(4), gripper(1) ]`,`quat = wxyz(T_G_ee[:3,:3])`,gripper 当前 0/1。
5. 维护 `n_obs=2` 观测窗(初始 `[obs0]*2`,每 chunk 滚动)。

> EE_OFFSET=0.10 与训练一致(`build_gt_replay.py:54`)。approach 符号:DP3 ez(腕→pinch)与 partner approach(R_ee→pinch)同向 → 减 0.10 都是"往后退向法兰",一致。

---

## 6. Razor 端逻辑:输出 retarget + IK + 驱动

DP3 吐 `action [8,8]`,每帧 = `[pos_G(3), quat_G_wxyz(4), gripper(1)]`(**绝对**位姿,物体系)。逐帧:

1. **物体系 → base**:`T_base_ee_dp3 = T_base_mesh @ T_G_ee`(`T_G_ee` 由 action 的 pos+quat 组装)。
2. **DP3 EE → pinch**:`T_base_pinch = T_base_ee_dp3`,原点 `+= 0.10·approach`(approach=列2),朝向不变。**(§5 步骤2 的逆)**
3. **pinch → R_ee**:`T_base_Ree = base_ee_from_virtual_pinch_closed(T_base_pinch, T_ee_pinch_closed)`（`hand_retarget_geometry.py:139`,**复用**;`T_ee_pinch_closed` 来自 `ee_retarget.yaml`,不重标)。
4. **IK**:`PinkLocalIK.solve_ik(T_base_Ree)`（`teleop/ik_utils.py:58`,**复用**),局部解、用当前 q 作种子 → 帧间关节连续(避免肘翻)。返回右臂 7-DOF。
5. **gripper**:二值;策略首次吐 `gripper≥0.5` → 触发 Sharpa `close_hand_until_stall`（`demo/hand_close.py:56`,复用);否则保持张开 `right_hand_profile.yaml`。
6. **驱动(平滑见 §7)**:把 `{left_arm(保持HOME), right_arm(IK解), left_hand, right_hand}` 写入 `action_buffer`,300Hz 线程负责限速平滑流给真机。
7. **闭环/终止**:每 8 个 waypoint 执行完 → 回 §5 重算 proprio → 下一 chunk。`object dz>0.03m` 且已闭手 → 成功早停;`max_chunks` 或策略始终不闭手 → 停。

---

## 7. 关节平滑 / 安全(⚠️ 真机保护,复用现成模块)

DP3 输出"一段段轨迹",chunk 间关节会跳;**不能直接发给真机**。V2AP-demo **已有现成模块**,不用自己写:

`teleop/arm_hand_control.py`:
- **300Hz 后台 `full_robot_action_loop`**(`executor.py:105` 启动):每 tick 读 `action_buffer` 的 4 个关节目标 → `smooth_and_check_action` → 发真机。
- **`SmoothingAndSafetyManager.smooth_and_check_action`(`:769`)**,每 tick:
  - **速度限幅平滑(默认)**(`:848-857`):`pos_diff = target − previous_smoothed`,逐关节 clip 到 `±DEXMATE_VEL_LIMIT_SCALE(0.4)·DEXMATE_DEFAULT_ARM_VEL_LIMITS / 300`。→ 即使 DP3 给个大跳变,**每 tick 只走一小步,平滑逼近**,不会突跳损坏真机。
  - (可选 **Ruckig** 高阶 jerk-limited,`:824`,但 demo 关闭 `ruckig_smoothing=False`,因免费版不支持 tracking。)
  - **关节限位 clip**(`:859-864`)。
  - **自碰撞检查**(`:866-874`):平滑后若自碰撞 → 回退上一个安全目标。
- **跟踪安全急停**(agent 报告):measured 与 target 偏差 > `TRACKING_SAFETY_THRESHOLD`(10.0 rad)→ 清 buffer + `terminate_event` 杀线程。粗 sanity cap;DP3 IK 解是合法关节配置,正常不会触发——但**务必帧间 IK 连续(种子=当前q)**,避免肘翻造成大跳。

> 用法:DP3 闭环只需在 chunk 频率(几 Hz)往 `action_buffer` 写 IK 关节目标(`executor._publish_action_buffer` / `apply_live_targets`,`:243`/`:214`,**复用**),300Hz 线程自动限速平滑。**左臂必须始终非 None**(loop 要求 4 键齐全)→ 保持其 HOME。

---

## 8. 复用 / 新建清单

| 模块 | 复用 partner? | 出处 |
|---|---|---|
| capture / T1–T5(感知全链) | ✅ 完全复用 | Titan demo/scripts |
| 点云采样(4096,no center/norm/rotate) | ✅ 复用采样代码 | `tools/infer_mesh_v6.py` + `glb_to_pdm_grasp.py` |
| `T_base_mesh` / base 对齐 mesh | ✅ 复用 | `T5/mesh_align.py`, `register/T_base_mesh.json` |
| pinch 帧 FK(proprio EE) | ✅ 复用 | `virtual_pinch_frame_in_base` |
| pinch→R_ee retarget | ✅ 复用 | `base_ee_from_virtual_pinch_closed` + `ee_retarget.yaml` |
| Pink IK | ✅ 复用 | `teleop/ik_utils.py PinkLocalIK` |
| 平滑/安全/驱动(300Hz) | ✅ 复用 | `arm_hand_control.py` + `action_buffer` |
| Sharpa stall-close | ✅ 复用 | `demo/hand_close.py` |
| DP3 推理 server | ✅ 复用 sim 版 | `Baseline1/eval/dp3_inference_server.py` |
| **T6_dp3**(采点云+起server+导session) | ❌ 新建(Titan) | `demo/scripts/T6_dp3/` |
| **run_auto_grasp_dp3 + dp3_client**(闭环驱动) | ❌ 新建(Razor) | `demo/phase2/` |
| **DP3 EE↔pinch 0.10 偏移**(§5.2/§6.2) | ❌ 新建(常量) | 两端各一行 |

---

## 9. 待与 partner 对齐 / 确认项

1. **⚠️ pinch 手指约定不统一(已查实,真实风险)**:部署 checkpoint = `dexycb162 + oakink207` 混合。
   - DexYCB 半:`build_gt_replay.py` 硬编码 **拇指+食指**(`j[4],j[8]`,thumb_index/rigid_body 两模式都用这俩)。
   - OakInk 半:源 retarget 目录 `episodes_oakink_v3_*pinch-middle*` 用 **拇指+中指**。
   - 影响:pinch 中点 + finger_open 轴随手指选择而变(食指 vs 中指对捏,中点差 ~1–1.5cm,轴略偏)。**不破坏逻辑链**——只是 §5 `virtual_pinch_frame_in_base` 用 Sharpa 哪两指的参数。
   - **决定(已与用户确认)**:部署**统一用拇指+食指**(`right_thumb_DP`/`right_index_DP`),理由:① 与 partner `ee_retarget.yaml` 标定一致(复用前提);② 与主方法虚拟二指夹爪一致;③ 与 DexYCB 半一致。
   - **历史背景**:曾训过一版**全程拇指+食指**的 DP3;后因 OakInk 拇指+食指采成功 episode 太少,改成"DexYCB 不变 + OakInk 改拇指+中指"重训了**混合版**(= 当前 A/B eval 用的 `dexycb162+oakink207` epoch2800,SR 25.1%)。
   - **checkpoint 选择(不阻塞开发,部署端代码相同,仅换 `--ckpt`)**:
     - 混合版:OakInk 半部署有 ~1cm 系统偏移,但 OakInk 覆盖全;
     - 全食指版:train/deploy 约定零偏移、最干净,但 OakInk 训练数据少、覆盖弱。
     - 选哪版看"想 demo 哪些物体 + 哪版 SR 更好",真机试或对比 eval 后再定。
   - retarget-mode(thumb_index vs rigid_body):两模式 EE pos 都是 `pinch−0.10·approach`、approach=列2,pinch 回退一致;仅朝向参数化不同,部署端 FK 给出物理 pinch 朝向即可,不需区分。
2. **HOME 关节(已定方向)**:先用 partner `DEFAULT_JOINT_POS` / start 位姿 → FK 得 state0;**真机效果不好再调**(潜在调参方向,记下)。
3. **base 重力对齐(已确认 ✅)**:Dexmate base Z = 垂直向上 = 与重力反向 = world up。G 系 = base 对齐 mesh 系成立,base↔G 纯平移。
4. **EE_OFFSET=0.10(已定 ✅)**:DP3 路径全程用训练值 0.10,不混主方法 panda_hand TCP 0.105。
5. **传输(已定)**:Razor SSH 公钥加入 server;proprio 每 chunk 走 HTTP `/predict`(Razor 直连 server),mesh/点云走一次性 session rsync。Server 选型见 §10。

---

## 10. Server 选型(5090 vs A6000)

DP3 推理 server **很轻**:~2GB 显存、单次 `/predict` ~几十 ms。**选型不取决于算力**(两台都绰绰有余、延迟差异相对每 chunk 8-waypoint 执行时间可忽略),取决于三点:① 雷蛇网络可达;② demo 时不被其他任务抢;③ DP3 python 环境 + checkpoint 在哪。

| | RTX 5090(本机) | A6000 |
|---|---|---|
| DP3 env + checkpoint + 部署代码 | ✅ 已在此 | ❌ 需复制环境+权重 |
| 当前占用 | GraspVLA 16h eval(~26GB/77%)+ 你的 RL,**有争用** | 未知(本机 hostname 解析不了,需你那边 host 配置) |
| 建议 | **v0 默认用 5090**(零搭建,demo 时确保 GraspVLA 跑完/RL 暂停即可) | 若需 demo 期间 5090 完全空出、或 A6000 物理更靠近机器人/同子网,则迁过去(一次性搭 env+拷 ckpt) |

**结论**:v0 用 **5090**(东西都在这、最快起);若你确认 A6000 可达且更适合常驻 demo,我再把 DP3 env + checkpoint 迁过去。
