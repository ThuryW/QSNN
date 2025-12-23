#!/bin/bash

# ================= 配置区域 =================
# 通用配置
GPUS="0,1,2"
DATASET="cifar10"
DATA_DIR="/home/wangtianyu/dataset"
ARCH="resnet20"

# 缩放因子 Tau
# ==== 最重要的两个参数τ =====
TAU=2
TAU_L0=10
# =========================

# 权重位宽和时间步配置
WEIGHT_BITS=2
TIMESTEPS=4

# 权重路径
CHECKPOINT="/home/wangtianyu/QSNN/checkpoints/spk_${ARCH}_w${WEIGHT_BITS}_t${TIMESTEPS}.pth.tar"
OUT_DIR="./export_int/w${WEIGHT_BITS}_tau_${TAU}_taul0_${TAU_L0}"


SCRIPT="scripts/export/export_int.py"

# ==========================================
echo ">>> Exporting Integer Model & Stats"
echo ">>> Output Dir: $OUT_DIR"
echo ">>> TAU: $TAU | TAU_L0: $TAU_L0 | WEIGHT_BITS: $WEIGHT_BITS"

# 构建基础命令
CMD="python $SCRIPT \
  -g $GPUS \
  -dd $DATA_DIR \
  -cpt $CHECKPOINT \
  -t $TIMESTEPS \
  --tau $TAU \
  --tau_l0 $TAU_L0 \
  --out_dir $OUT_DIR \
  -wb $WEIGHT_BITS"

# 执行命令
echo ">>> 执行命令:"
echo "$CMD"
eval $CMD