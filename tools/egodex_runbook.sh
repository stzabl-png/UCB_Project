# EgoDex 全量处理 Runbook
# 环境: hawor (MANO/接触图), bundlesdf (FoundationPose)
# 工作目录: /home/lyh/Project/Affordance2Grasp
cd /home/lyh/Project/Affordance2Grasp

# ═══════════════════════════════════════════════════════════════
# STEP 0: 帧提取（mp4 → extracted_images/）
# 全部 3051 条序列，fps=5，4 workers 并行
# 估计时间: ~3-4 小时 (取决于视频时长)
# ═══════════════════════════════════════════════════════════════
python tools/batch_extract_egodex.py --fps 5 --workers 4

# 只跑某个任务（测试用）:
# python tools/batch_extract_egodex.py --task add_remove_lid


# ═══════════════════════════════════════════════════════════════
# STEP 1: MegaSAM — 相机位姿 + Metric Depth
# 对每条序列运行 MegaSAM，输出 cam_c2w.npy + depth.npz + K.npy
# MegaSAM 需要单独的环境，命令参考已有序列的处理方式
# 输出: data_hub/ProcessedData/egocentric_depth/egodex/{task}/{subj}/
#
# 参考指令 (MegaSAM 环境内):
#   for seq in data_hub/RawData/.../egodex/test/*/*/extracted_images/; do
#       task=$(echo $seq | cut -d/ -f...); subj=...
#       python /path/to/MegaSAM/run.py --input $seq --output egocentric_depth/egodex/$task/$subj
#   done
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# STEP 2: HaWoR — MANO 手部估计（world space）
# 对每条序列运行 HaWoR，输出 world_space_res.pth
# 需要 MegaSAM 的 SLAM 位姿作为输入
# 输出: data_hub/RawData/.../egodex/test/{task}/{subj}/world_space_res.pth
#
# HaWoR 批量指令参考 (hawor env):
#   conda run -n hawor python /path/to/hawor/demo.py \
#       --video_path extracted_images/ \
#       --slam_path egocentric_depth/egodex/{task}/{subj}/ \
#       --output_path .../{task}/{subj}/
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# STEP 3: 【手动，每个新物体做一次】SAM2 标注 + SAM3D 重建
#
# 3a. 用 SAM2 交互式工具标注物体 mask：
#     → 保存到 data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png
#     → 保存原图到 data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/image.png
#
# 3b. SAM3D 点云重建（上传到 SAM3D 云服务）：
#     → 下载 mesh，保存到 data_hub/ProcessedData/obj_meshes/egocentric/{obj}/mesh.ply
#
# EgoDex 主要物体 (101个任务，需逐步标注):
#   assemble_tile, lego, jenga_block, card, book, ...
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# STEP 4: 注册新序列（每新增一个任务/物体都要更新这3个文件）
#
# 1. tools/batch_obj_pose_ego.py  → SEQUENCE_REGISTRY
# 2. tools/gen_ego_contact_map.py → SEQUENCE_REGISTRY
# 3. data/estimate_obj_scale_ego.py → OBJ_DEPTH_MAP
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# STEP 5: Scale 估计（每个物体）
# ═══════════════════════════════════════════════════════════════
conda run -n hawor python data/estimate_obj_scale_ego.py
# 全部已注册物体: (不加 --obj 就跑所有)
# 输出: obj_meshes/egocentric/{obj}/scale.json


# ═══════════════════════════════════════════════════════════════
# STEP 6: FoundationPose 物体位姿估计
# ═══════════════════════════════════════════════════════════════
conda run -n bundlesdf python tools/batch_obj_pose_ego.py --dataset egodex
# 输出: obj_poses_ego/egodex/{seq}/ob_in_cam/*.txt


# ═══════════════════════════════════════════════════════════════
# STEP 7: 接触图生成
# ═══════════════════════════════════════════════════════════════
conda run -n hawor python tools/gen_ego_contact_map.py
# 全部注册物体，跑完输出:
# data_hub/human_prior/{obj}.hdf5


# ═══════════════════════════════════════════════════════════════
# STEP 8: 可视化验证
# ═══════════════════════════════════════════════════════════════
conda run -n hawor python tools/vis_ego_contact.py --obj assemble_tile
eog output/affordance_ego/assemble_tile_contact_3d.png

# ═══════════════════════════════════════════════════════════════
# 当前已完成状态（可以直接从 STEP 5 开始）
# ═══════════════════════════════════════════════════════════════
# ✅ assemble_disassemble_tiles/1 → 帧已提取(1089帧), MegaSAM done, HaWoR done
# ✅ assemble_tile → SAM2 mask, SAM3D mesh, scale.json 均已准备
# ✅ assemble_tile → FP poses (60帧), contact map (8%), vis 已完成
#
# ❌ 其余 3050 条序列 → 需要从 STEP 0 开始（已有 mp4，等待帧提取）
# ❌ 其余 ~100 个物体 → 需要 STEP 3 手动标注
