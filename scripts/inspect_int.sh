#!/bin/bash

# ================= 通用配置 =================
GPUS="0,1,2"
DATA_DIR="/home/wangtianyu/dataset"
ARCH="resnet20"         # resnet20 或 vgg11
DATASET="cifar10"
BATCH_SIZE=256

# ==== 最重要的两个参数τ =====
TAU=2
TAU_L0=10
# =========================

# 权重位宽和时间步
WEIGHT_BITS=2
TIMESTEPS=4

# 权重路径
CHECKPOINT="/home/wangtianyu/QSNN/checkpoints/spk_${ARCH}_w${WEIGHT_BITS}_t${TIMESTEPS}.pth.tar"

# log文件目录
LOG_DIR="./logs_inspect"

# inpect脚本名
SCRIPT_NAME="inspect_int.py"

# ========================================
echo ">>> Starting SNN Inspection ($MODE Mode)..."
echo ">>> Arch: $ARCH | Tau: $TAU | Tau_L0: $TAU_L0 | Weight_Bits: $WEIGHT_BITS"
echo ">>> Mode: $MODE | Script: $SCRIPT_NAME"

export PYTHONPATH=$PYTHONPATH:$(pwd)

CMD="python scripts/inspect/$SCRIPT_NAME \
    -g $GPUS \
    -cpt $CHECKPOINT \
    -dd $DATA_DIR \
    -a $ARCH \
    -d $DATASET \
    -t $TIMESTEPS \
    -b $BATCH_SIZE \
    --tau $TAU \
    --tau_l0 $TAU_L0 \
    --log_dir $LOG_DIR \
    -wb $WEIGHT_BITS"

# 执行命令
echo ">>> 执行命令:"
echo "$CMD"
eval $CMD