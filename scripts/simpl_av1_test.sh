CUDA_VISIBLE_DEVICES=0 python test.py \
  --features_dir data_argo/features_new/ \
  --train_batch_size 4 \
  --mode test \
  --val_batch_size 4 \
  --use_cuda \
  --adv_cfg_path cfg.simpl_cfg_yaml \
  --model_path saved_models/20250419-221600/Simpl_epoch32.tar