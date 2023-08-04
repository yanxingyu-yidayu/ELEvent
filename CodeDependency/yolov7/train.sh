#!/bin/bash
#SBATCH -J yolo                #任务名称
#SBATCH -N 1                  #节点数
#SBATCH --ntasks-per-node=8  #核心数
#SBATCH -p kshdtest           #队列名称
#SBATCH --gres=dcu:1
source /public/home/ac2ax5rex7/miniconda3/etc/profile.d/conda.sh

conda activate yolo  #激活环境
module switch compiler/rocm/dtk-22.04.1
cd /public/home/ac2ax5rex7/projects/lift/yolov7-main

python -u train.py --weights weights/yolov7_training.pt --cfg cfg/training/yolov7-lift.yaml --data data/lift.yaml --device 0  --batch-size 8 --epoch 300


