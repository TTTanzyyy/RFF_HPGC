#!/bin/bash
GPU_ID=0
N_TRIALS=1 # number of experimental trials to run

echo " >>> Running cifar10 simulation!"
DATA_DIR="data/cifar10/"
OUT_DIR="exp/pff/cifar10/"
python src/sim_train_1_3.py --data_dir=$DATA_DIR --gpu_id=$GPU_ID --n_trials=$N_TRIALS --out_dir=$OUT_DIR

