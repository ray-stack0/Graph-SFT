CUDA_VISIBLE_DEVICES=0 python test.py \
  --features_dir data_av2/features/ \
  --train_batch_size 32 \
  --mode test \
  --val_batch_size 32 \
  --use_cuda \
  --adv_cfg_path cfg.simpl_av2_cfg \
  --model_path saved_models/20250718-011132/Simpl_ddp_best.tar