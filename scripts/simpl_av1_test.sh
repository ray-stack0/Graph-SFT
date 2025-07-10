CUDA_VISIBLE_DEVICES=0 python test.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --mode test \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path cfg.simpl_cfg \
  --model_path saved_models/20250708-220436_Simpl_epoch50.tar