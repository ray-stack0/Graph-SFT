CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 train_ddp.py \
  --features_dir data_argo/features/ \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --val_interval 2 \
  --train_epoches 60 \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path cfg.simpl_cfg \
  --experiment_name Point-RPE-GAT-MapEncoder+RPEGAT-GCNv2-Fusion
  # --resume \
  # --model_path saved_models/20250619-012346/Simpl_ddp_ckpt_epoch30.tar