CUDA_VISIBLE_DEVICES=0 python test.py \
  --features_dir data_av2/features/ \
  --train_batch_size 16 \
  --mode test \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path cfg.simpl_av2_cfg \
  --model_path saved_models/20250715-104220/Simpl_best.tar