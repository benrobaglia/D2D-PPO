import numpy as np
import torch
import pickle
from envs.channel_selection_env import ChannelSelectionEnv
from algorithms.d2d_ppo import D2DPPO
from algorithms.ippo import iPPO

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))

def time_to_slot(t):
    Tf_gf = 4*(1 / 30 * 1e-3 + 2.34e-6)
    return t / Tf_gf

n_seeds = 1
n_agents = 6
# ts = np.array([0.5e-3, 1e-3, 1.5e-3, 2e-3])
# inter_arrival_list = time_to_slot(ts)
n_channels = 32
loads = [1/14, 1/6, 1/3, 1]

ippo_scores_list = []
ippo_jains_list = []
ippo_channel_errors_list = []
ippo_average_rewards_list = []
training_list = []

for seed in range(n_seeds):
    print(f"Seed {seed}")
    ippo_scores_list_seed = []
    ippo_jains_list_seed = []
    ippo_channel_errors_list_seed = []
    ippo_average_rewards_list_seed = []
    training_list_seed = []

    for load in loads:
        print(f"load= {load}")
        deadlines = np.array([7] * n_agents)
        channel_switch = np.array([0.5 for _ in range(n_channels+1)])
        lbdas = np.array([load for _ in range(n_agents)])
        period = None
        arrival_probs = None
        offsets = None

        env = ChannelSelectionEnv(n_agents=n_agents,
                              n_channels=n_channels,
                              deadlines=deadlines,
                              lbdas=lbdas,
                              episode_length=200,
                              traffic_model="aperiodic",
                              arrival_probs=None,
                              offsets=None,
                              channel_switch=channel_switch,
                             verbose=False)
        
        # d2dppo = D2DPPO(env,
        #         hidden_size=64, 
        #         gamma=0.5, 
        #         policy_lr=1e-4,
        #         value_lr=1e-3,
        #         device=None,
        #         useRNN=True,
        #         history_len=10,
        #         early_stopping=True)
        
        # res = d2dppo.train(num_iter=5000, num_episodes=20, n_epoch=4, test_freq=100)    
        # score_ippo, jains_ippo, channel_error_ippo, rewards_ippo = d2dppo.test(500)

        ippo = iPPO(env, useRNN=True, history_len=10)
        res = ippo.train(num_iter=2500, n_epoch=4, num_episodes=10, test_freq=100)    
        score_ippo, jains_ippo, channel_error_ippo, rewards_ippo = ippo.test(500)

        print(f"URLLC score iPPO: {score_ippo}")
        print(f"Jain's index iPPO: {jains_ippo}")
        print(f"Channel errors iPPO: {channel_error_ippo}")
        print(f"Reward per episode iPPO: {rewards_ippo}\n")

    training_list_seed.append(res)
    training_list.append(training_list_seed)

    ippo_scores_list_seed.append(score_ippo)
    ippo_jains_list_seed.append(jains_ippo)
    ippo_channel_errors_list_seed.append(channel_error_ippo)
    ippo_average_rewards_list_seed.append(rewards_ippo)
    
    ippo_scores_list.append(ippo_scores_list_seed)
    ippo_jains_list.append(ippo_jains_list_seed)
    ippo_channel_errors_list.append(ippo_channel_errors_list_seed)
    ippo_average_rewards_list.append(ippo_average_rewards_list_seed)



d2dppo_result = {"scores": ippo_scores_list,
                "jains": ippo_jains_list, 
                "channel_errors": ippo_channel_errors_list, 
                "average_rewards": ippo_average_rewards_list,
                "xp_params": {'loads': loads, 'deadlines': 7},
                "training": training_list
               }


pickle.dump(d2dppo_result, open('results/ippo_load_xp.p', 'wb'))