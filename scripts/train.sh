#!/bin/bash

torchrun \
    --module \
    engine.main_train \
    -g 0,1,2 \
    -a resnet20 \
    -p 32 \
    -s \
    -t 5 \
    -d cifar10 \
    -e 300 \
    -lr 0.04 \
    -ld ./log \
    -dd /home/wangtianyu/dataset