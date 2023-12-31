import numpy as np
import torch
import pickle
from envs.channel_selection_env import ChannelSelectionEnv
from algorithms.d2d_ppo import D2DPPO
from algorithms.ippo import iPPO

path = 'results/xp_gamma_ippo.p'

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))
print(f"path: {path}")

def time_to_slot(t):
    Tf_gf = 4*(1 / 30 * 1e-3 + 2.34e-6)
    return t / Tf_gf

n_seeds = 1
n_agents = 5
# ts = np.array([0.5e-3, 1e-3, 1.5e-3, 2e-3])
# inter_arrival_list = time_to_slot(ts)
n_channels = 16
# loads = [1/21, 1/14, 1/7, 1/3.5 1/1.75]
load = 1/3.5
gammas = [0.2, 0.4, 0.6, 0.8, 0.99]

ppo_scores_list = []
ppo_jains_list = []
ppo_channel_errors_list = []
ppo_average_rewards_list = []
training_list = []


for gamma in gammas:
    print(f"gamma= {gamma}")
    deadlines = np.array([7] * n_agents)
    channel_switch = np.array([0.8 for _ in range(n_channels+1)])
    lbdas = np.array([load for _ in range(n_agents)])
    period = np.array([7 for _ in range(n_agents)])
    arrival_probs = np.array([1 for _ in range(n_agents)])
    offsets = np.array([0, 2, 4, 0, 2])
    periodic_devices = np.array([2, 4])

    env = ChannelSelectionEnv(n_agents=n_agents,
                            n_channels=n_channels,
                            deadlines=deadlines,
                            period=period,
                            lbdas=lbdas,
                            episode_length=200,
                            traffic_model="aperiodic",
                            arrival_probs=arrival_probs,
                            periodic_devices=periodic_devices,
                            offsets=offsets,
                            channel_switch=channel_switch,
                            verbose=False)
    
    # d2dppo = D2DPPO(env,
    #         hidden_size=64, 
    #         gamma=0.4, 
    #         policy_lr=3e-4,
    #         value_lr=1e-2,
    #         device=None,
    #         useRNN=True,
    #         history_len=10,
    #         early_stopping=True)
    
    # res = d2dppo.train(num_iter=1500, num_episodes=10, n_epoch=4, test_freq=100)    
    # score_ppo, jains_ppo, channel_error_ppo, rewards_ppo = d2dppo.test(500)

    ippo = iPPO(env, 
                    hidden_size=64, 
                    gamma=gamma,
                    policy_lr=3e-4,
                    value_lr=1e-2,
                    device=None,
                    useRNN=True,
                    history_len=10,
                    early_stopping=True
                    )

    res = ippo.train(num_iter=1000, n_epoch=4, num_episodes=10, test_freq=100)    
    score_ppo, jains_ppo, channel_error_ppo, rewards_ppo = ippo.test(500)


    print(f"URLLC score ppo: {score_ppo}")
    print(f"Jain's index ppo: {jains_ppo}")
    print(f"Channel errors ppo: {channel_error_ppo}")
    print(f"Reward per episode ppo: {rewards_ppo}\n")

    training_list.append(res)

    ppo_scores_list.append(score_ppo)
    ppo_jains_list.append(jains_ppo)
    ppo_channel_errors_list.append(channel_error_ppo)
    ppo_average_rewards_list.append(rewards_ppo)
    


ppo_result = {"scores": ppo_scores_list,
                "jains": ppo_jains_list, 
                "channel_errors": ppo_channel_errors_list, 
                "average_rewards": ppo_average_rewards_list,
                "xp_params": {'gammas': gammas, 'deadlines': 7},
                "training": training_list
               }


pickle.dump(ppo_result, open(path, 'wb'))