import numpy as np
import torch
import pickle
from envs.env import D2DEnv
from algorithms.ippo import iPPO

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))

n_seeds = 1
n_agents_list = [2, 4, 6, 8, 10]

ippo_scores_list = []
ippo_jains_list = []
ippo_channel_errors_list = []
ippo_average_rewards_list = []

for seed in range(n_seeds):
    print(f"Seed {seed}")
    ippo_scores_list_seed = []
    ippo_jains_list_seed = []
    ippo_channel_errors_list_seed = []
    ippo_average_rewards_list_seed = []

    for k in n_agents_list:
        print(f"k={k}")
        deadlines = np.array([7] * k)
        lbdas = np.array([1/14] * k)
        period = None
        arrival_probs = None
        offsets = None
        # neighbourhoods = [list(range(k)) for i in range(k)] # neighbourhoods full obs
        neighbourhoods = [[i] for i in range(k)]
        print(f"Neighbourhoods: {neighbourhoods}")

        env = D2DEnv(k,
                    deadlines,
                    lbdas,
                    period=period,
                    arrival_probs=arrival_probs,
                    offsets=offsets,
                    episode_length=200,
                    traffic_model='aperiodic',
                    periodic_devices=[],
                    reward_type=0,
                    channel_switch=0,
                    channel_decoding=1.,
                    neighbourhoods=neighbourhoods, # Neighbourhoods is a list of size n_agents with the indices of the neighbours for each agent.
                    verbose=False)
        
        ippo = iPPO(env)
        res = ippo.train(num_iter=5000, num_episodes=10, test_freq=500)    
        score_ippo, jains_ippo, channel_error_ippo, rewards_ippo = ippo.test(500)

        print(f"URLLC score iPPO: {score_ippo}")
        print(f"Jain's index iPPO: {jains_ippo}")
        print(f"Channel errors iPPO: {channel_error_ippo}")
        print(f"Reward per episode iPPO: {rewards_ippo}\n")

    ippo_scores_list_seed.append(score_ippo)
    ippo_jains_list_seed.append(jains_ippo)
    ippo_channel_errors_list_seed.append(channel_error_ippo)
    ippo_average_rewards_list_seed.append(rewards_ippo)
    
    ippo_scores_list.append(ippo_scores_list_seed)
    ippo_jains_list.append(ippo_jains_list_seed)
    ippo_channel_errors_list.append(ippo_channel_errors_list_seed)
    ippo_average_rewards_list.append(ippo_average_rewards_list_seed)


ippo_result = {"ippo_scores": ippo_scores_list, "ippo_jains": ippo_jains_list, "ippo_channel_errors": ippo_channel_errors_list, "ippo_average_rewards": ippo_average_rewards_list,
                       "xp_params": {'n_agents_list': n_agents_list, 'deadlines': 7, 'interarrivals':14, 'neighbourhoods': neighbourhoods}
                       }


pickle.dump(ippo_result, open('results/ippo_partial_obs.p', 'wb'))