# Run locally
set NUM_CPUS=4
set RAY_ROOT=D:/ray
python experiment_binary.py config/local/b1_f01234.json D:/mice_perf_local/f01234 --temp_folder D:/mice_perf_tmp/f01234 --epochs 20
