#!/bin/bash
#sleep 4000
echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir /host_home/ltp/Dataset/AV1/val/data/ \
  --save_dir data_argo/features/
sleep 30
echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir /host_home/ltp/Dataset/AV1/train/data/ \
  --save_dir data_argo/features/
sleep 30
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 train_ddp.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --val_interval 3 \
  --train_epoches 50 \
  --use_cuda \
  --data_aug \
  --logger_writer \
  --adv_cfg_path cfg.simpl_cfg \
  --experiment_name av1-0725
