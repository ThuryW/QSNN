#!/bin/bash

# ================= 通用配置区域 =================
# 1. 硬件
GPUS="0,1,2"                      # 指定 GPU，例如 "0" 或 "0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPUS

# 2. 数据
DATASET="cifar10"                # cifar10, cifar100, imagenet
DATA_DIR="/home/wangtianyu/dataset"

# 3. 模型
ARCH="vgg11"                  # resnet20, vgg11

# 4. 评估模式: "q" 或 "fp"
MODE="q"                     # 修改此变量切换评估模式

# ================= 模式特定配置 =================

if [ "$MODE" = "q" ]; then
    # ================= 量化评估配置 =================
    
    # 量化参数
    BITS=2
    TIMESTEPS=4
    
    # 权重文件路径 (必须指定)
    CHECKPOINT="/home/wangtianyu/QSNN/checkpoints/spk_${ARCH}_w${BITS}_t${TIMESTEPS}.pth.tar"
    
    # 批量大小
    BATCH_SIZE=256
    
    # SNN 参数
    SPIKING=true
    
    echo "运行量化评估: Bits=${BITS}, Timesteps=${TIMESTEPS}"
    echo "模型: $ARCH (Spiking: $SPIKING)"
    echo "权重: $CHECKPOINT"
    
    # 运行量化评估
    python3 scripts/eval_test/eval_quant.py \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --arch $ARCH \
        --precision $BITS \
        --timesteps $TIMESTEPS \
        --resume $CHECKPOINT \
        --batch_size $BATCH_SIZE

elif [ "$MODE" = "fp" ]; then
    # ================= 全精度评估配置 =================
    
    # 时间步 (仅在 SPIKING=true 时生效)
    TIMESTEPS=4
    
    # 类型设置：如果是 SNN 模型，设为 true；如果是 ANN，设为 false
    SPIKING=false
    
    # 权重文件路径 (必须是全精度权重)
    CHECKPOINT="/home/wangtianyu/quant_snn/quant_dist/checkpoints/${ARCH}_fp.pth.tar"
    
    # 批量大小
    BATCH_SIZE=128
    
    # 计算 GPU 数量
    NUM_PROCS=$(echo $GPUS | tr ',' '\n' | wc -l)
    
    # 基础参数
    CMD_ARGS="-g $GPUS -a $ARCH -d $DATASET -dd $DATA_DIR -b $BATCH_SIZE -cpt $CHECKPOINT"
    
    # 追加 SNN 参数
    if [ "$SPIKING" = true ]; then
        CMD_ARGS="$CMD_ARGS -s -t $TIMESTEPS"
    fi
    
    echo "正在启动全精度评估..."
    echo "模型: $ARCH (Spiking: $SPIKING)"
    echo "权重: $CHECKPOINT"
    
    # 使用 torchrun 启动
    torchrun \
        --nproc_per_node=$NUM_PROCS \
        --master_port=29501 \
        scripts/eval_test/eval_accuracy.py $CMD_ARGS

else
    echo "错误: 未知的评估模式 '$MODE'。请使用 'quant' 或 'fp'。"
    exit 1
fi