#aim is to test whether exploration vs reward makes a different on sparse cartpole swingup
pip install git+https://github.com/Bmillidgework/exploration-baselines
#python main.py --logdir "log_sparse_cartpole_exploration" --use_exploration "True" --env_name "SparseCartpoleSwingup" --render "True"
python main.py --logdir "log_sparse_cartpole_reward_only" --use_exploration "False" --env_name "SparseCartpoleSwingup" --render "True" --action_noise 0.3
# so it looks like it can perfectly solve cartpole after about 100 epochs. Would love to get a reward curve out of that.
#not sure what other experiments I should be running. Key will be actually understanding the tabular case properly including with the code.
