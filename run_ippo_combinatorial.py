import numpy as np
import torch
import pickle
from envs.combinatorial_env import CombinatorialEnv
from algorithms.ippo import iPPO
import os
import random

# Fix random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


xp_name = 'combinatorial_load'
output_path = f'{xp_name}/results/ippo.p'

if xp_name not in os.listdir():
    print(f"Creating directories for experiment...")
    os.mkdir(f'{xp_name}')
    os.mkdir(f'{xp_name}/results')

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))
print(f"Launching {xp_name} experiment...")
print(f"output_path: {output_path}")


n_seeds = 1
n_agents = 5
n_channels = 10
# loads = [1/14, 1/7, 1/3.5, 1/1.75, 1, 1.25]
loads = [1.25]


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

    for load in loads:
        print(f"load= {load}")
        # Managing directories for models
        model_folder = "models_ippo"
        if model_folder not in os.listdir(xp_name):
            os.mkdir(f"{xp_name}/{model_folder}")

        deadlines = np.array([7, 10, 14, 17, 20])
        channel_switch = np.array([0.8 for _ in range(n_channels)])
        lbdas = np.array([load for _ in range(n_agents)])
        # period = np.array([7 for _ in range(n_agents)])
        # arrival_probs = np.array([1 for _ in range(n_agents)])
        # offsets = np.array([0, 2, 4, 0, 2])
        # periodic_devices = np.array([2, 4])

        
        env = CombinatorialEnv(n_agents=n_agents,
                                n_channels=n_channels,
                                deadlines=deadlines,
                                lbdas=lbdas,
                                episode_length=200,
                                traffic_model='aperiodic',
                                channel_switch=channel_switch,
                                verbose=False)

        ippo = iPPO(env, 
                    hidden_size=64, 
                    gamma=0.99,
                    policy_lr=3e-4,
                    value_lr=1e-2,
                    device=None,
                    useRNN=True,
                    save_path=f"{xp_name}/{model_folder}",
                    combinatorial=True,
                    history_len=10,
                    early_stopping=True
                    )
        
        res = ippo.train(num_iter=2000, n_epoch=4, num_episodes=10, test_freq=100)    

        ippo.load(f"{xp_name}/{model_folder}")
        score_ppo, jains_ppo, channel_error_ppo, rewards_ppo = ippo.test(500)

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
                "xp_params": {'loads': loads, 'deadlines': 7},
                "env": env, # Save the env to save all parameters
                "training": training_list
               }


pickle.dump(ppo_result, open(output_path, 'wb'))