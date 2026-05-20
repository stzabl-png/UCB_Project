#!/bin/bash
# check_ready.sh — 检查 Pipeline 是否准备完毕

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
OK="✅"; FAIL="❌"; WAIT="⏳"

PROJ=/home/lyh/Project/Affordance2Grasp
SPROJ=/home/vision/Project/Affordance2Grasp
SERVER=vision@128.32.164.115
PD=$PROJ/data_hub/ProcessedData
ALL_OK=1

chk() { [ "$2" = "1" ] && echo -e "$OK  $1" || { echo -e "$FAIL  $1"; ALL_OK=0; }; }
waitt() { echo -e "$WAIT  $1 (进行中)"; ALL_OK=0; }

echo ""
echo "════════════════════════════════════════════"
echo "       Pipeline 准备状态检查"
echo "════════════════════════════════════════════"

# ── 本机 ─────────────────────────────────────────
echo ""
echo "【本机 — EgoDex Pipeline】"

# HaPTIC 下载
haptic_done=0
[ -f "$PROJ/third_party/haptic/output/release/mix_all/checkpoints/last.ckpt" ] && haptic_done=1
if [ $haptic_done -eq 1 ]; then
  chk "HaPTIC 模型权重" 1
else
  part=$(ls $PROJ/third_party/haptic/output/haptic_model.tar.gz* 2>/dev/null | head -1)
  if [ -n "$part" ]; then
    size=$(du -sh "$part" 2>/dev/null | cut -f1)
    waitt "HaPTIC 下载中 (已下 $size / 11GB)"
  else
    chk "HaPTIC 模型权重" 0
  fi
fi

chk "HaWoR 权重" $([ -f "$PROJ/third_party/hawor/weights/hawor/checkpoints/finetuned/hawor_finetuned_final.ckpt" ] && echo 1 || echo 0)
chk "MegaSAM 权重" $([ -f "$PROJ/mega-sam/checkpoints/megasam_final.pth" ] && echo 1 || echo 0)
chk "EgoDex Mask" $([ "$(find $PD/mask/Egocentric/Egodex -name '*.png' 2>/dev/null | wc -l)" -gt 10 ] && echo 1 || echo 0)
chk "EgoDex RawMesh" $([ -d "$PD/RawMesh/meshes/egodex" ] && echo 1 || echo 0)
chk "EgoDex RawData" $([ "$(ls $PROJ/data_hub/RawData/EgoRawData/egodex/ 2>/dev/null | wc -l)" -gt 0 ] && echo 1 || echo 0)

# ── 服务器 ────────────────────────────────────────
echo ""
echo "【Titan 服务器 — OakInk + DexYCB Pipeline】"

SERVER_STATUS=$(ssh $SERVER "
  mask_ok=\$(find $SPROJ/data_hub/ProcessedData/mask/ThirdPerson/oakink -name '*.png' 2>/dev/null | wc -l)
  mesh_ok=\$([ -d '$SPROJ/data_hub/ProcessedData/RawMesh/meshes/oakink' ] && echo 1 || echo 0)
  fp_ok=\$([ -f '$SPROJ/third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth' ] && echo 1 || echo 0)
  raw_oak=\$(ls '$SPROJ/data_hub/RawData/ThirdPersonRawData/oakink_v1/' 2>/dev/null | wc -l)
  raw_dex=\$(ls '$SPROJ/data_hub/RawData/ThirdPersonRawData/dexycb/' 2>/dev/null | wc -l)
  depth_ok=\$([ -f '$SPROJ/third_party/ml-depth-pro/checkpoints/depth_pro.pt' ] && echo 1 || echo 0)
  haptic_ok=\$([ -f '$SPROJ/third_party/haptic/output/release/mix_all/checkpoints/last.ckpt' ] && echo 1 || echo 0)
  echo \"\$mask_ok|\$mesh_ok|\$fp_ok|\$raw_oak|\$raw_dex|\$depth_ok|\$haptic_ok\"
" 2>/dev/null)

if [ -z "$SERVER_STATUS" ]; then
  echo -e "$FAIL  无法连接服务器"
  ALL_OK=0
else
  IFS='|' read -r mask_n mesh_ok fp_ok raw_oak raw_dex depth_ok haptic_ok <<< "$SERVER_STATUS"
  chk "OakInk Mask ($mask_n 物体)" $([ "$mask_n" -gt 50 ] && echo 1 || echo 0)
  chk "RawMesh 已同步" "$mesh_ok"
  chk "FoundationPose 权重" "$fp_ok"
  chk "OakInk RawData ($raw_oak 序列)" $([ "$raw_oak" -gt 100 ] && echo 1 || echo 0)
  chk "DexYCB RawData ($raw_dex 受试者)" $([ "$raw_dex" -gt 0 ] && echo 1 || echo 0)
  chk "DepthPro 权重" "$depth_ok"
  if [ "$haptic_ok" = "1" ]; then
    chk "HaPTIC 模型权重" 1
  else
    waitt "HaPTIC 权重未同步 (等本机下载完后 rsync)"
  fi
fi

# ── 后台任务 ─────────────────────────────────────
echo ""
echo "【后台任务状态】"
pgrep -f "rsync.*mask" > /dev/null    && waitt "mask 同步进行中"    || echo -e "$OK  mask 同步已完成/未运行"
pgrep -f "rsync.*RawMesh" > /dev/null && waitt "RawMesh 同步进行中" || echo -e "$OK  RawMesh 同步已完成/未运行"
pgrep -f "rsync.*Foundation" > /dev/null && waitt "FP 同步进行中"   || echo -e "$OK  FP 同步已完成/未运行"
pgrep -f "gdown\|dl_model" > /dev/null  && waitt "HaPTIC 下载进行中 ($(du -sh $PROJ/third_party/haptic/output/haptic_model.tar.gz* 2>/dev/null | tail -1 | cut -f1) / 11GB)" || echo -e "$OK  HaPTIC 下载已完成/未运行"

# ── 最终结论 ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
if [ $ALL_OK -eq 1 ]; then
  echo -e "${GREEN}✅  全部就绪！可以开始运行 Pipeline${NC}"
  echo "   服务器: bash run_pipeline_gpu0_oakink.sh &"
  echo "          bash run_pipeline_gpu1_dexycb.sh &"
  echo "   本机:   bash run_pipeline_ego_egodex.sh"
else
  echo -e "${YELLOW}⏳  尚未完全就绪，等待上述 ⏳ 项完成后再运行${NC}"
fi
echo "════════════════════════════════════════════"
echo ""
