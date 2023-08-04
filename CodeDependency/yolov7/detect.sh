#!/bin/bash
#SBATCH -J yolov7detect                #任务名称
#SBATCH -N 2                  #节点数
#SBATCH --ntasks-per-node=32  #核心数
#SBATCH -p kshdtest           #队列名称
#SBATCH --gres=dcu:4
source /public/home/ac2ax5rex7/miniconda3/etc/profile.d/conda.sh

export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DEBUG_CONV_WINOGRAD=0 
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export HSA_USERPTR_FOR_PAGED_MEM=0
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export MIOPEN_SYSTEM_DB_PATH=/temp/pytorch-miopen-2.8


module switch compiler/rocm/dtk-22.04.1
conda activate yolo  #激活环境
cd /public/home/ac2ax5rex7/projects/lift/yolov7-main

python detect3.py --weights runs/train/exp13/weights/best.pt --source /public/home/ac2ax5rex7/datasets/demoVideo/elm --save-txt --save-conf