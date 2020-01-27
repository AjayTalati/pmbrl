# comparing standard reward only vs exploration on spare cartpole

pip install git+https://github.com/Bmillidgework/exploration-baselines

python main.py --logdir "log_cartpole_expl_1" --use_exploration "True" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False"
python main.py --logdir "log_cartpole_expl_2" --use_exploration "True" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False"
python main.py --logdir "log_cartpole_expl_3" --use_exploration "True" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False"
python main.py --logdir "log_cartpole_expl_4" --use_exploration "True" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False"
python main.py --logdir "log_cartpole_expl_5" --use_exploration "True" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False"

python main.py --logdir "log_cartpole_reward_1" --use_exploration "False" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3"
python main.py --logdir "log_cartpole_reward_2" --use_exploration "False" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3"
python main.py --logdir "log_cartpole_reward_3" --use_exploration "False" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3"
python main.py --logdir "log_cartpole_reward_4" --use_exploration "False" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3"
python main.py --logdir "log_cartpole_reward_5" --use_exploration "False" --env_name "SparseCartpoleSwingup"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --save_model "False" --action_noise = "0.3"
