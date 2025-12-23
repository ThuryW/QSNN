#!/bin/bash

# ================= 配置区域 =================

# 1. 硬件
GPUS="0,1,2,3"                  # 指定 GPU，例如 "0" 或 "0,1,2,3"

# 2. 数据
DATASET="cifar10"         # cifar10, cifar100, imagenet
DATA_DIR="/home/wangtianyu/dataset"

# 3. 模型
ARCH="resnet20"           # resnet20, vgg11
BATCH_SIZE=128

# 4. 类型设置
# 如果是 SNN 模型，设为 true；如果是 ANN，设为 false
SPIKING=false
TIMESTEPS=3               # SNN 时间步 (仅在 SPIKING=true 时生效)

# 5. 权重文件路径 (必须是全精度权重)
CHECKPOINT="/home/wangtianyu/quant_snn/quant_dist/checkpoints/resnet20_fp.pth.tar"


# ================= 执行区域 =================

# 计算 GPU 数量
NUM_PROCS=$(echo $GPUS | tr ',' '\n' | wc -l)

# 基础参数
CMD_ARGS="-g $GPUS -a $ARCH -d $DATASET -dd $DATA_DIR -b $BATCH_SIZE -cpt $CHECKPOINT"

# 追加 SNN 参数
if [ "$SPIKING" = true ]; then
    CMD_ARGS="$CMD_ARGS -s -t $TIMESTEPS"
fi

echo ">>> 正在启动全精度评估..."
echo ">>> 模型: $ARCH (Spiking: $SPIKING)"
echo ">>> 权重: $CHECKPOINT"

# 使用 torchrun 启动
torchrun \
    --nproc_per_node=$NUM_PROCS \
    --master_port=29501 \
    scripts/eval_test/eval_accuracy.py $CMD_ARGS