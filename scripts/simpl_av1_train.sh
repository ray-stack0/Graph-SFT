ulimit -n 4096  # 增加文件描述符限制到4096
CUDA_VISIBLE_DEVICES=0 python train.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --val_interval 2 \
  --train_epoches 60 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path cfg.simpl_cfg \
  --experiment_name EdgeGAT*3+GlobalQueryRefine+aWTA+aWTA_ade