export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
echo "-- Processing AV2 val set..."
python data_av2/run_preprocess.py --mode val \
  --data_dir /home/nvidia/ltp/Dataset/AV2/val/ \
  --save_dir data_av2/features/ \
  --small
# --debug --viz

echo "-- Processing AV2 train set..."
python data_av2/run_preprocess.py --mode train \
  --data_dir /home/nvidia/ltp/Dataset/AV2/train/ \
  --save_dir data_av2/features/ \
  --small

# echo "-- Processing AV2 test set..."
# python data_av2/run_preprocess.py --mode test \
#   --data_dir /home/nvidia/ltp/Dataset/AV2/ \
#   --save_dir data_av2/features/ \
#   --small