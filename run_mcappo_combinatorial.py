import numpy as np
import torch
import pickle
from envs.combinatorial_env import CombinatorialEnv
from algorithms.d2d_ppo import D2DPPO

path = 'results/mcappo_combinatorial_load_xp.p'

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))
print(f"path: {path}")

def time_to_slot(t):
    Tf_gf = 4*(1 / 30 * 1e-3 + 2.34e-6)
    return t / Tf_gf

n_seeds = 1
n_agents = 5
# ts = np.array([0.5e-3, 1e-3, 1.5e-3, 2e-3])
# inter_arrival_list = time_to_slot(ts)
n_channels = 8
loads = [1/21, 1/14, 1/7, 1/3.5, 1/1.75]

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
        deadlines = np.array([7] * n_agents)
        channel_switch = np.array([0.8 for _ in range(n_channels+1)])
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

        ppo = D2DPPO(env, 
                        hidden_size=64, 
                        gamma=0.99,
                        policy_lr=3e-4,
                        value_lr=1e-2,
                        device=None,
                        useRNN=True,
                        combinatorial=True,
                        history_len=10,
                        early_stopping=True
                        )
        
        res = ppo.train(num_iter=1000, n_epoch=4, num_episodes=5, test_freq=100)    
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
                "xp_params": {'loads': loads, 'deadlines': 7},
                "training": training_list
               }


pickle.dump(ppo_result, open(path, 'wb'))