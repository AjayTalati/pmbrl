#run mountain car a bunch of different times to get reasonable benchmark
pip install git+https://github.com/Bmillidgework/exploration-baselines
#python main.py --logdir "log_mc_reward_infogain_only_1" --use_exploration "False" --env_name "SparseMountainCar" --render "False"  --max_episode_len 200 --n_episodes 100 --use_reward_info_gain "True" --use_ensemble_reward_model "True"
python main.py --logdir "log_mc_reward_infogain_only_2" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --use_reward_info_gain "True" --use_ensemble_reward_model "True"
python main.py --logdir "log_mc_reward_infogain_only_3" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --use_reward_info_gain "True" --use_ensemble_reward_model "True"
python main.py --logdir "log_mc_reward_infogain_only_4" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --use_reward_info_gain "True" --use_ensemble_reward_model "True"
python main.py --logdir "log_mc_reward_infogain_only_5" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --use_reward_info_gain "True" --use_ensemble_reward_model "True"

#mountain car exploration
python main.py --logdir "log_mc_expl_1" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_expl_2" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_expl_3" --use_exploration "True" --env_name "SparseMountainCar"  --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_expl_4" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_expl_5" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100

# reward only
python main.py --logdir "log_mc_reward_only_1" --use_exploration "False" --env_name "SparseMountainCar"  --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_reward_only_2" --use_exploration "False" --env_name "SparseMountainCar"  --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_reward_only_3" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_reward_only_4" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100
python main.py --logdir "log_mc_reward_only_5" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100
