# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn

from .measures import InformationGain


class CEMPlanner(nn.Module):
    def __init__(
        self,
        ensemble,
        reward_model,
        action_size,
        plan_horizon,
        optimisation_iters,
        n_candidates,
        top_candidates,
        use_reward=True,
        use_exploration=True,
        use_reward_info_gain=False,
        expl_scale=1,
        return_stats = True,
        device="cpu",
    ):
        super().__init__()
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.action_size = action_size
        self.ensemble_size = ensemble.ensemble_size

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates

        self.use_reward = use_reward
        self.use_exploration = use_exploration
        self.expl_scale = expl_scale
        self.use_reward_info_gain = use_reward_info_gain
        self.device = device

        self.reward_list = []
        self.measure = InformationGain(self.ensemble)
        self.return_stats = return_stats

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.state_size = state.size(0)
        state = state.unsqueeze(dim=0).unsqueeze(dim=0)
        state = state.repeat(self.ensemble_size, self.n_candidates, 1)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )

        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            states, delta_vars, delta_means = self.perform_rollout(state, actions)
            returns = torch.zeros(self.n_candidates).float().to(self.device)

            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                expl_bonus = expl_bonus.sum(dim=0)
                returns += expl_bonus

            if self.use_reward:
                if self.reward_model.ensemble_reward_model:
                    states = states.view(self.ensemble_size, -1, self.state_size)
                else:
                    states = states.view(-1, self.state_size)
                rewards = self.reward_model(states)

                self.rewards = rewards.view(
                    self.plan_horizon, self.ensemble_size, self.n_candidates
                )

                rewards = self.rewards.mean(dim=1).sum(dim=0)
                self.reward_list.append(rewards)
                returns += rewards

            if self.use_reward_info_gain:
                self.rewards = self.rewards.unsqueeze(3)
                reward_info_gain = torch.zeros([self.plan_horizon, self.n_candidates]).to(self.device)
                for t in range(self.plan_horizon):
                  reward_info_gain[t,:] = self.measure.entropy_of_average(self.rewards[t,:,:])
                #print("reward_info_gain: ", torch.sum(reward_info_gain,dim=0))
                returns += torch.sum(reward_info_gain,dim=0)

            if self.return_stats:
                self.reward_list.append(rewards)
                if self.use_exploration:
                    self.info_list.append(expl_bonus)


            returns = torch.where(
                torch.isnan(returns), torch.zeros_like(returns), returns
            )

            _, topk = returns.topk(
                self.top_candidates, dim=0, largest=True, sorted=False
            )

            best_actions = actions[:, topk.view(-1)].reshape(
                self.plan_horizon, self.top_candidates, self.action_size
            )

            action_mean, action_std_dev = (
                best_actions.mean(dim=1, keepdim=True),
                best_actions.std(dim=1, unbiased=False, keepdim=True),
            )

        if self.return_stats:
            reward_stats, info_stats = self.get_stats()
            return action_mean[0].squeeze(dim=0), reward_stats, info_stats

        return action_mean[0].squeeze(dim=0)

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T
        states[0] = current_state

        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var)
            # states[t + 1] = mean
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means

    def get_stats(self):
        self.reward_list = torch.stack(self.reward_list).view(-1)
        self.info_list = torch.stack(self.info_list).view(-1) * self.expl_scale
        reward_stats = {
            "max": self.reward_list.max().item(),
            "mean": self.reward_list.mean().item(),
            "min": self.reward_list.min().item(),
            "std": self.reward_list.std().item(),
        }
        if self.use_exploration:
            info_stats = {
                "max": self.info_list.max().item(),
                "mean": self.info_list.mean().item(),
                "min": self.info_list.min().item(),
                "std": self.info_list.std().item(),
            }
        else:
            info_stats = {}
        self.info_list = []
        seoflreward_list = []
        return reward_stats, info_stats


### Path Integral Planner ###
class PIPlanner(nn.Module):
    def __init__(
        self,
        dynamics_model,
        reward_model,
        action_size,
        N_samples,
        plan_horizon,
        lambda_,
        noise_mu,
        noise_sigma,
        env,
        use_exploration=True,
        use_reward=True,
        use_reward_info_gain = False,
        device="cpu"
    ):
        super().__init__()
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.action_size = action_size
        self.N_samples = N_samples
        self.plan_horizon = plan_horizon
        self.ensemble_size = self.dynamics_model.ensemble_size
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma=noise_sigma
        self.use_exploration = use_exploration
        self.use_reward = use_reward
        self.use_reward_info_gain = use_reward_info_gain
        self.device = device
        self.action_trajectory= torch.zeros([self.plan_horizon, self.action_size]).to(self.device)
        self.env=copy.deepcopy(env)
        self.times_called = 0

    def state_from_obs(self, state):
        theta = np.arccos(state[0])
        return np.array([theta, state[2]])

    def rollout(self, current_state,noise):
        """
        Performs rollout with the transition model forwards in time. Current state is size [State_dim]
        """
        T = self.plan_horizon +1
        states = [torch.empty(0)] * T
        info_gains = (
            torch.zeros(self.plan_horizon, self.N_samples).float().to(self.device)
        )
        ent_avgs = (
            torch.zeros(self.plan_horizon, self.N_samples).float().to(self.device)
        )
        avg_ents = (
            torch.zeros(self.plan_horizon, self.N_samples).float().to(self.device)
        )
        #repeat current state ensemble and sample times
        """Map current state to [Ensemble_size, N_samples, State_dim]"""
        current_state = current_state.unsqueeze(0).unsqueeze(0).repeat(self.ensemble_size,self.N_samples,1)
        states[0] = current_state
        #repeat actions and states for multiple samples
        """Map action trajectory to [Ensemble_size, N_samples, Action_dim, 1]"""
        actions = self.action_trajectory.unsqueeze(0).unsqueeze(0).repeat(self.ensemble_size,self.N_samples,1,1)
        """ Half the actions are the previous trajectory. Half start from scratch to enable exploration """
        actions[:,self.N_samples//2:self.N_samples-1,:,:] = torch.zeros([self.ensemble_size,(self.N_samples//2)-1,self.plan_horizon, self.action_size])

        for t in range(self.plan_horizon):
            mean, var = self.dynamics_model(states[t], actions[:,:,t,:] + noise[:,:,t,:])
            states[t+1] = self.dynamics_model.sample(mean, var)
            if self.use_exploration:
              ent_avg = entropy_of_average(states[t+1])
              avg_ent = average_of_entropy(var)
              info_gains[t, :] = ent_avg - avg_ent
              ent_avgs[t, :] = ent_avg
              avg_ents[t, :] = avg_ent
        """ Turn list into a torch tensor and sum across time """
        states = torch.stack(states[1:], dim=0)
        """ Predict the rewards from the states """
        costs = torch.sum(-self.reward_model(states), dim=0)
        #print("costs in rollout:", costs.size())
        logger._reward_list.append(torch.sum(costs,dim=0).cpu())
        if self.use_exploration:
          info_gains = torch.sum(info_gains, dim=0)
          #print("info gains: ", info_gains.size())
          ent_avgs = torch.sum(ent_avgs, dim=0)
          avg_ents = torch.sum(avg_ents, dim=0)
          logger._info_gains.append(info_gains.cpu())
          logger._ent_avgs.append(ent_avgs.cpu())
          logger._avg_ents.append(avg_ents.cpu())
          info_gains = info_gains.unsqueeze(0).repeat(self.ensemble_size,1).unsqueeze(2)
          #print("adjusted info gains: ", info_gains.size())
          ### CHECK!!! COSTS MINUS INFO GAINS SINCE COSTS ARE - REWARDS?
          return states, costs - info_gains

        if self.use_reward_info_gain:
            raise(NotImplementedError("This isn't done yet!"))

        return states, costs

    def real_env_rollout(self, current_state, noise):
        current_state = current_state.cpu()
        noise = noise.cpu()
        costs = torch.zeros([self.ensemble_size, self.N_samples,self.action_size])
        for j in range(self.ensemble_size):
            for k in range(self.N_samples):
                s = self.env.reset()
                self.env._env.state = self.state_from_obs(current_state.numpy())
                for t in range(self.plan_horizon):
                    action = self.action_trajectory[t].cpu() + noise[j,k,t,:]
                    s, reward, _ = self.env.step(action)
                    costs[j,k,:] += reward.cpu()
        return None, costs.to(self.device)


    def SG_filter(self, action_trajectory):
        WINDOW_SIZE = 5
        POLY_ORDER = 3
        return torch.tensor(signal.savgol_filter(action_trajectory, WINDOW_SIZE,POLY_ORDER,axis=0))

    def forward(self, current_state):
        noise = torch.randn([self.ensemble_size, self.N_samples, self.plan_horizon,self.action_size]) * self.noise_sigma
        noise = noise.to(self.device)
        """ costs: [Ensemble_size, N_samples] """
        states, costs = self.rollout(current_state,noise)
        costs = costs /torch.mean(torch.sum(torch.abs(costs),dim=1)) #normalize first here might be the easiest way of getting this sorted. then I can adjust params around
        #print("costs: ", costs[0,1:10,:])
        """ beta is for numerical stability. Aim is that all costs before negative exponentiation are small and around 1 """
        beta = torch.min(costs)
        costs = torch.exp(-(1/self.lambda_ * (costs - beta)))
        eta = torch.mean(torch.sum(costs,dim=1)) + 1e-10
        """ weights: [Ensemble_size, N_samples] """
        weights = (1/eta) * costs
        #print("weights: ", weights[0,1:10,:])
        """ Multiply weights by noise and sum across time dimension """
        #print("noise: ", noise[0,1:10,0,:])
        #print("multiplied: ", weights[0,1:10,:] * noise[0,1:10,0,:])

        add = torch.stack([torch.sum(torch.sum(weights * noise[:,:,t,:],dim=1),dim=0) for t in range(self.plan_horizon)])
        self.times_called+=1
        if self.times_called >= 100:
          print("costs: ", costs[0,1:10,:])
          print("weights: ", weights[0,1:10,:])
          print("add: ", add)
          self.times_called = 0
        self.action_trajectory += self.SG_filter(add.cpu()).to(self.device)
        action = self.action_trajectory[0] * 5
        """ Move forward action trajectory by 1 in preparation for next time-step """
        self.action_trajectory = torch.roll(self.action_trajectory,-1)
        self.action_trajectory[self.plan_horizon-1] = 0
        #print('action: ',action.item())
        return action



### Random shooting planner ###
# Just samples a bunch of trajectories at random and picks the best one.

class RandomShootingPlanner(nn.Module):
    def __init__(self, dynamics_model,reward_model, action_size,plan_horizon, N_samples, action_noise_sigma, use_exploration, use_reward, discount_factor=1,device='cpu'):
        self.dynamics_model = dynamics_model
        self.ensemble_size = self.dynamics_model.ensemble_size
        self.reward_model = reward_model
        self.action_size = action_size
        self.plan_horizon = plan_horizon
        self.N_samples = N_samples,
        self.action_noise_sigma = action_noise_sigma
        self.use_reward = use_reward
        self.use_exploration = use_exploration
        self.discount_factor = discount_factor
        self.device=device
        if self.discount_factor <1:
            self.discount_factor_matrix = self._initialize_discount_factor_matrix()

    def _initialize_discount_factor_matrix(self):
        discounts = np.zeros([self.plan_horizon,1,1,1])
        for t in range(self.plan_horizon):
            discounts[t,:,:,:] = self.discount_factor ** self.plan_horizon
        discounts=torch.from_numpy(discounts).repeat(1,self.ensemble_size, self.N_samples, 1).to(self.device)
        return discounts

    def forward(self, current_state):
        self.state_size = current_state.size(0)
        state = state.unsqueeze(dim=0).unsqueeze(dim=0)
        state = state.repeat(self.ensemble_size, self.n_candidates, 1)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        states, delta_vars, delta_means = self.perform_rollout(state, actions)
        returns = torch.zeros(self.n_candidates).float().to(self.device)

        if self.use_exploration:
            expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
            expl_bonus = expl_bonus.sum(dim=0)
            returns += expl_bonus

        if self.use_reward:
            if self.reward_model.ensemble_reward_model:
                states = states.view(self.ensemble_size, -1, self.state_size)
            else:
                states = states.view(-1, self.state_size)
            rewards = self.reward_model(states)

            rewards = rewards.view(
                self.plan_horizon, self.ensemble_size, self.n_candidates
            )

            rewards = rewards.mean(dim=1).sum(dim=0)
            returns += rewards

        if self.discount_factor <1:
            returns *= self.discount_factor_matrix

        returns = torch.where(
            torch.isnan(returns), torch.zeros_like(returns), returns
        )

        _, topk = returns.topk(
            1, dim=0, largest=True, sorted=True
        )

        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, 1, self.action_size
        )
        return best_actions[0,:,:]
