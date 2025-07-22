# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
echo "-- Processing AV2 val set..."
python data_av2/run_preprocess.py --mode val \
  --data_dir /host_home/AV2/val/ \
  --save_dir data_av2/features/

echo "-- Processing AV2 train set..."
python data_av2/run_preprocess.py --mode train \
  --data_dir /host_home/AV2/train/ \
  --save_dir data_av2/features/

echo "-- Processing AV2 test set..."
python data_av2/run_preprocess.py --mode test \
  --data_dir /host_home/AV2/test \
  --save_dir data_av2/features/ 
