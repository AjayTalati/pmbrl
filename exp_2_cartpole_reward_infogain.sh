#reward information gain only for cartpole.
pip install git+https://github.com/Bmillidgework/exploration-baselines
python main.py --logdir "log_sparse_cartpole_reward_infogain_only" --use_exploration "False" --env_name "SparseCartpoleSwingup" --render "True" --use_reward_info_gain "True" --use_ensemble_reward_model "True" 
