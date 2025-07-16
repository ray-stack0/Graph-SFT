# 设置使用的GPU设备（0,1,2,3 四张卡）
# export CUDA_VISIBLE_DEVICES="3"
export MASTER_PORT=29531
ulimit -n 4096  # 增加文件描述符限制到4096
CUDA_VISIBLE_DEVICES=0
python opt_paramerters.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --val_interval 2 \
  --train_epoches 25 \
  --data_aug  \
  --use_cuda \
  --adv_cfg_path cfg.simpl_cfg \
  --is_main_thread 