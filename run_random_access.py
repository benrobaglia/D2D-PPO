import numpy as np
import torch
import pickle
from envs.channel_selection_env import ChannelSelectionEnv
from algorithms.baselines import RandomAccess

path = 'results/random_access_load.p'

print(f"path: {path}")


n_seeds = 1
n_agents = 5
n_channels = 16
loads = [1/14, 1/6, 1/3, 1/1.5, 1]
#loads = [1/1.5]

scores_list = []
jains_list = []
channel_errors_list = []
average_rewards_list = []

for seed in range(n_seeds):
    print(f"Seed {seed}")
    scores_list_seed = []
    jains_list_seed = []
    channel_errors_list_seed = []
    average_rewards_list_seed = []
    training_list_seed = []

    for load in loads:
        print(f"load= {load}")
        deadlines = np.array([7] * n_agents)
        channel_switch = np.array([0.8 for _ in range(n_channels+1)])
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
        
        ra = RandomAccess(env, True)
        score, jains, channel_error, rewards = ra.run(500)

        print(f"URLLC score: {score}")
        print(f"Jain's index: {jains}")
        print(f"Channel errors: {channel_error}")
        print(f"Reward per episode: {rewards}\n")

        scores_list_seed.append(score)
        jains_list_seed.append(jains)
        channel_errors_list_seed.append(channel_error)
        average_rewards_list_seed.append(rewards)
    
    scores_list.append(scores_list_seed)
    jains_list.append(jains_list_seed)
    channel_errors_list.append(channel_errors_list_seed)
    average_rewards_list.append(average_rewards_list_seed)



result = {"scores": scores_list,
                "jains": jains_list, 
                "channel_errors": channel_errors_list, 
                "average_rewards": average_rewards_list,
                "xp_params": {'loads': loads, 'deadlines': 7},
               }


pickle.dump(result, open(path, 'wb'))

