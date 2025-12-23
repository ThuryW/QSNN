#!/bin/bash

torchrun \
    --module \
    engine.main_dist \
    -g 0,1,2 \
    -a resnet20 \
    -d cifar10 \
    -ts \
    -tp 32 \
    -tt 4 \
    -ss \
    -sp 2 \
    -st 4 \
    -e 50 \
    -lr 2e-4 \
    -ld ./log \
    -dd /home/wangtianyu/dataset \
    -cpt /home/wangtianyu/quant_snn/quant_dist/checkpoints/spk_resnet20_fp_t4.pth.tar