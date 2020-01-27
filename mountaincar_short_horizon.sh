pip install git+https://github.com/Bmillidgework/exploration-baselines

python main.py --logdir "log_ar1_mc_expl_short_horizon_1" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_expl_short_horizon_2" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_expl_short_horizon_3" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_expl_short_horizon_4" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_expl_short_horizon_5" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --plan_horizon "12"

python main.py --logdir "log_ar1_mc_reward_short_horizon_1" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_reward_short_horizon_2" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_reward_short_horizon_3" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_reward_short_horizon_4" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --plan_horizon "12"
python main.py --logdir "log_ar1_mc_reward_short_horizon_5" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --plan_horizon "12"
