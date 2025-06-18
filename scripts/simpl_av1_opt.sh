# 设置使用的GPU设备（0,1,2,3 四张卡）
export CUDA_VISIBLE_DEVICES="3"
export MASTER_PORT=29534
python opt_paramerters.py \
  --features_dir data_argo/features/ \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --val_interval 2 \
  --train_epoches 6 \
  --use_cuda \
  --adv_cfg_path cfg.simpl_cfg