#aim is to test whether acrobot reward is vital for that or not.
#I also need to do mountain car with action repeat = 1, same I think with acrobot...
#although that might alread have it

pip install git+https://github.com/Bmillidgework/exploration-baselines
python main.py --logdir "log_acrobot_exploration_1" --use_exploration "True" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_acrobot_exploration_2" --use_exploration "True" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_acrobot_exploration_3" --use_exploration "True" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_acrobot_exploration_4" --use_exploration "True" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_acrobot_exploration_5" --use_exploration "True" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0"

python main.py --logdir "log_acrobot_exploration_1" --use_exploration "False" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_acrobot_exploration_2" --use_exploration "False" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_acrobot_exploration_3" --use_exploration "False" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_acrobot_exploration_4" --use_exploration "False" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_acrobot_exploration_5" --use_exploration "False" --env_name "SparseDoublePendulum"   --max_episode_len 500 --n_episodes 100  --action_repeat "1" --env_std "0.0" --action_noise "0.3"

# test also mountain car with action repeat = 1

python main.py --logdir "log_ar1_mc_expl_1" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_ar1_mc_expl_2" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_ar1_mc_expl_3" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_ar1_mc_expl_4" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0"
python main.py --logdir "log_ar1_mc_expl_5" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0"

python main.py --logdir "log_ar1_mc_reward_1" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_ar1_mc_reward_2" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_ar1_mc_reward_3" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_ar1_mc_reward_4" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3"
python main.py --logdir "log_ar1_mc_reward_5" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3"
