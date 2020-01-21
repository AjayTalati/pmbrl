#aim is to test whether exploration vs reward makes a different on sparse cartpole swingup
pip install git+https://github.com/Bmillidgework/exploration-baselines
python main.py --logdir "log_sparse_cartpole_exploration" --use_exploration "True" --env_name "SparseCartpoleSwingup"
python main.py --logdir "log_sparse_cartpole_exploration" --use_exploration "False" --env_name "SparseCartpoleSwingup"
