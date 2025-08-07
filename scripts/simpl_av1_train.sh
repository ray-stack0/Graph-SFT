ulimit -n 4096  # 增加文件描述符限制到4096
CUDA_VISIBLE_DEVICES=3 python train.py \
  --features_dir data_argo/features/ \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --val_interval 4 \
  --train_epoches 50 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path cfg.simpl_cfg \
  --experiment_name av1-M3.2