
pip install git+https://github.com/Bmillidgework/exploration-baselines

python main.py --logdir "log_acrobot_expl_large_ensemble_1" --use_exploration "True" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_acrobot_expl_large_ensemble_2" --use_exploration "True" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_acrobot_expl_large_ensemble_3" --use_exploration "True" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_acrobot_expl_large_ensemble_4" --use_exploration "True" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_acrobot_expl_large_ensemble_5" --use_exploration "True" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"

python main.py --logdir "log_acrobot_reward_large_ensemble_1" --use_exploration "False" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3" --ensemble_size "20"
python main.py --logdir "log_acrobot_reward_large_ensemble_2" --use_exploration "False" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3" --ensemble_size "20"
python main.py --logdir "log_acrobot_reward_large_ensemble_3" --use_exploration "False" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3" --ensemble_size "20"
python main.py --logdir "log_acrobot_reward_large_ensemble_4" --use_exploration "False" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3" --ensemble_size "20"
python main.py --logdir "log_acrobot_reward_large_ensemble_5" --use_exploration "False" --env_name "SparseDoublePendulume"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3" --ensemble_size "20"
