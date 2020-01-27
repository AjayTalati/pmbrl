# aim here is to try out large ensemble sizes on mountain car w/ action repeat = 1. To see if this helps
#ensemble size of 20, for a bigger ensemble size, which seems reasonabe.

pip install git+https://github.com/Bmillidgework/exploration-baselines

python main.py --logdir "log_ar1_mc_expl_large_ensemble_1" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_expl_large_ensemble_2" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_expl_large_ensemble_3" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_expl_large_ensemble_4" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_expl_large_ensemble_5" --use_exploration "True" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --save_model "False" --ensemble_size "20"

python main.py --logdir "log_ar1_mc_reward_large_ensemble_1" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_reward_large_ensemble_2" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_reward_large_ensemble_3" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_reward_large_ensemble_4" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --save_model "False" --ensemble_size "20"
python main.py --logdir "log_ar1_mc_reward_large_ensemble_5" --use_exploration "False" --env_name "SparseMountainCar"   --max_episode_len 200 --n_episodes 100 --action_repeat "1" --env_std "0.0" --action_noise "0.3" --save_model "False" --ensemble_size "20"
