#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

# 1. 指向第一步训练好的权重
PRETRAINED_PATH="/home/wangtianyu/QSNN/checkpoints/spk_resnet20_fp_t4.pth.tar"

# 2. 设置 precision 为 2 (极低比特)
# 3. 减小学习率 (0.01 或 0.005)
# 4. 使用 --resume 加载权重
torchrun \
    --module \
    engine.main_train \
    --gpus 0,1,2 \
    --arch resnet20 \
    --dataset cifar10 \
    -dd /home/wangtianyu/dataset \
    -e 100 \
    --learning_rate 0.01 \
    --spiking \
    --precision 4 \
    --timesteps 4 \
    --log_dir ./log/quant_finetune \
    --resume $PRETRAINED_PATH