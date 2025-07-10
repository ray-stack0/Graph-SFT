# echo "-- Processing val set..."
# python data_argo/run_preprocess.py --mode val \
#   --data_dir /home/nvidia/ltp/Dataset/Argoverse1/val/data/ \
#   --save_dir data_argo/features/

# echo "-- Processing train set..."
# python data_argo/run_preprocess.py --mode train \
#   --data_dir /home/nvidia/ltp/Dataset/Argoverse1/train/data/ \
#   --save_dir data_argo/features/

echo "-- Processing test set..."
python data_argo/run_preprocess.py --mode test \
  --data_dir /home/nvidia/ltp/Dataset/Argoverse1/test_obs/data/ \
  --save_dir data_argo/features/