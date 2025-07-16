CUDA_VISIBLE_DEVICES=0 python test.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --mode test \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path cfg.simpl_cfg \
  --model_path saved_models/20250714-100714/Simpl_best.tar