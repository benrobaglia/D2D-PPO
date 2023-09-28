import numpy as np
import torch
import pickle
from envs.combinatorial_env import CombinatorialEnv
from algorithms.d2d_ppo import D2DPPO
from algorithms.ippo import iPPO
import os
import random

# Fix random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

xp_name = 'xp_3gpp_homogeneous'
output_path = f'{xp_name}/results/ippo.p'

if xp_name not in os.listdir():
    print(f"Creating directories for experiment...")
    os.mkdir(f'{xp_name}')
    os.mkdir(f'{xp_name}/results')

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))
print(f"path: {output_path}")

def time_to_slot(t):
    Tf_gf = 4*(1 / 30 * 1e-3 + 2.34e-6)
    return t / Tf_gf

n_seeds = 1
# ts = np.array([0.5e-3, 1e-3, 1.5e-3, 2e-3])
# inter_arrival_list = time_to_slot(ts)
n_channels = 4
load = 1/14
n_agents_list = [4, 8, 12, 16]
# n_agents_list = [16]

ppo_scores_list = []
ppo_jains_list = []
ppo_channel_errors_list = []
ppo_average_rewards_list = []
training_list = []

for seed in range(n_seeds):
    print(f"Seed {seed}")
    ppo_scores_list_seed = []
    ppo_jains_list_seed = []
    ppo_channel_errors_list_seed = []
    ppo_average_rewards_list_seed = []
    training_list_seed = []

    for n_agents in n_agents_list:
        print(f"n_agents= {n_agents}")
        # model_folder = f"models_mcappo_seed_{seed}_k_{n_agents}"
        model_folder = f"models_ippo_seed_{seed}_k_{n_agents}"

        if model_folder not in os.listdir(xp_name):
            os.mkdir(f"{xp_name}/{model_folder}")
        
        deadlines = np.array([7] * n_agents)
        channel_switch = np.ones((n_agents, n_channels)) * 0.8
        lbdas = np.array([load for _ in range(n_agents)])
        # period = np.array([7 for _ in range(n_agents)])
        # arrival_probs = np.array([1 for _ in range(n_agents)])
        # offsets = np.array([0, 2, 4, 0, 2])
        # periodic_devices = np.array([2, 4])

        
        env = CombinatorialEnv(n_agents=n_agents,
                                n_channels=n_channels,
                                deadlines=deadlines,
                                lbdas=lbdas,
                                period=None,
                                arrival_probs=None,
                                offsets=None,
                                episode_length=200,
                                traffic_model='aperiodic',
                                collision_type='pessimistic',
                                periodic_devices=[],
                                channel_switch=channel_switch,
                                verbose=False)

        ppo = iPPO(env, 
                    hidden_size=64, 
                    gamma=0.4,
                    policy_lr=3e-4,
                    value_lr=1e-3,
                    device=None,
                    useRNN=True,
                    save_path=f"{xp_name}/{model_folder}",
                    combinatorial=True,
                    history_len=n_agents,
                    early_stopping=True
                    )

        # ppo = D2DPPO(env, 
        #         hidden_size=64, 
        #         gamma=0.4,
        #         policy_lr=3e-4,
        #         value_lr=1e-3,
        #         beta_entropy=0.01,
        #         device=None,
        #         useRNN=True,
        #         save_path=f"{xp_name}/{model_folder}",
        #         combinatorial=True,
        #         history_len=n_agents,
        #         early_stopping=True
        #         )

        
        res = ppo.train(num_iter=2000, n_epoch=5, num_episodes=15, test_freq=100)    
        ppo.load(f"{xp_name}/{model_folder}")
        score_ppo, jains_ppo, channel_error_ppo, rewards_ppo = ppo.test(500)

        print(f"URLLC score ppo: {score_ppo}")
        print(f"Jain's index ppo: {jains_ppo}")
        print(f"Channel errors ppo: {channel_error_ppo}")
        print(f"Reward per episode ppo: {rewards_ppo}\n")

        training_list_seed.append(res)
        training_list.append(training_list_seed)

        ppo_scores_list_seed.append(score_ppo)
        ppo_jains_list_seed.append(jains_ppo)
        ppo_channel_errors_list_seed.append(channel_error_ppo)
        ppo_average_rewards_list_seed.append(rewards_ppo)
    
    ppo_scores_list.append(np.array(ppo_scores_list_seed))
    ppo_jains_list.append(np.array(ppo_jains_list_seed))
    ppo_channel_errors_list.append(np.array(ppo_channel_errors_list_seed))
    ppo_average_rewards_list.append(np.array(ppo_average_rewards_list_seed))



ppo_result = {"scores": ppo_scores_list,
                "jains": ppo_jains_list, 
                "channel_errors": ppo_channel_errors_list, 
                "average_rewards": ppo_average_rewards_list,
                "xp_params": {'n_agents': n_agents_list, 'deadlines': 7},
                "training": training_list
               }


pickle.dump(ppo_result, open(output_path, 'wb'))