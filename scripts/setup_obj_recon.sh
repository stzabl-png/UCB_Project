#!/bin/bash
# ============================================================
#  setup_obj_recon.sh
#  一键安装 3D 物体重建模块（环境 + 模型权重）
#
#  用法:
#    bash scripts/setup_obj_recon.sh
#
#  完成后可运行:
#    conda activate obj-recon
#    python third_party/obj_recon/batch_infer.py --help
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OBJ_RECON_DIR="$PROJECT_DIR/third_party/obj_recon"
CKPT_TAG="hf"
CKPT_DIR="$OBJ_RECON_DIR/checkpoints/$CKPT_TAG"

echo "========================================"
echo " 3D Object Reconstruction Setup"
echo " Dir: $OBJ_RECON_DIR"
echo "========================================"

# ── Step 1: Conda 环境 ───────────────────────────────────────
echo ""
echo "[1/3] 创建 conda 环境 obj-recon ..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

if conda env list | grep -q "^obj-recon "; then
    echo "  ✅ 环境已存在，跳过创建"
else
    conda env create -n obj-recon \
        -f "$OBJ_RECON_DIR/environments/default.yml" \
        --no-default-packages 2>/dev/null || \
    conda create -n obj-recon python=3.11 -y
    echo "  ✅ conda 环境创建完成"
fi

# ── Step 2: pip 安装依赖 ─────────────────────────────────────
echo ""
echo "[2/3] 安装 Python 依赖 ..."
conda run -n obj-recon pip install \
    -r "$OBJ_RECON_DIR/requirements.inference.txt" \
    --quiet 2>&1 | tail -5

# 安装包本身（editable）
conda run -n obj-recon pip install -e "$OBJ_RECON_DIR" --quiet 2>&1 | tail -3
echo "  ✅ 依赖安装完成"

# ── Step 3: 下载模型权重 ────────────────────────────────────
echo ""
echo "[3/3] 下载模型权重 (facebook/sam-3d-objects, ~12GB) ..."
if [ -f "$CKPT_DIR/pipeline.yaml" ]; then
    echo "  ✅ 模型权重已存在，跳过下载"
else
    mkdir -p "$(dirname "$CKPT_DIR")"
    conda run -n obj-recon huggingface-cli download \
        --repo-type model \
        --local-dir "$CKPT_DIR-download" \
        --max-workers 2 \
        facebook/sam-3d-objects
    mv "$CKPT_DIR-download/checkpoints/$CKPT_TAG" "$CKPT_DIR" 2>/dev/null || \
    mv "$CKPT_DIR-download" "$CKPT_DIR"
    rm -rf "$CKPT_DIR-download"
    echo "  ✅ 模型权重下载完成"
fi

echo ""
echo "========================================"
echo " ✅ 安装完成！"
echo ""
echo " 测试: conda activate obj-recon"
echo "       cd $OBJ_RECON_DIR"
echo "       python demo.py"
echo ""
echo " 批量推理:"
echo "       python $PROJECT_DIR/third_party/obj_recon/batch_infer.py \\"
echo "         --input-dir $PROJECT_DIR/data_hub/ProcessedData/obj_recon_input \\"
echo "         --output-dir $PROJECT_DIR/data_hub/ProcessedData/obj_meshes"
echo "========================================"
