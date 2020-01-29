import json
import matplotlib.pyplot as plt
import sys
import numpy as np

class Results_plotter(object):
    def __init__(self, base_directory,N=1):
        self.base_directory = base_directory
        self.N = N
        if self.N ==1:
            self.results_dict = self._load_json(self.base_directory +"/metrics.json")
        else:
            self.results_dict_list = [self._load_json(self.base_directory + "_"+str(i+1)+"/metrics.json") for i in range(N)]


    def _load_json(self,path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

    def plot_reward_losses(self):
        if self.N == 1:
            reward_losses = self.results_dict['test_rewards']
            plt.plot(reward_losses)
            plt.show()
        else:
            reward_losses_list = [self.results_dict_list['test_rewards'] for i in range(N)]
            reward_losses_list = np.array(reward_losses_list)
            max_rloss = np.amax(reward_losses_list,axis=0)
            min_rloss = np.amin(reward_losses_list, axis=0)
            fig, axs = plt.subplots()
            for i in range(self.N):
                axs[0].plot(reward_losses_list[i,:])
            axs[0].fill_between(max_rloss, min_rloss,alpha='0.3')
            axs[0].set_title("Reward losses over " + str(self.N) + " seeds.")
            plt.show()



if __name__ == "__main__":
    rplot = Results_plotter("log_mc_reward_infogain_only_1", 1)
    rplot.plot_reward_losses()
