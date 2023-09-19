import numpy as np
import pickle
from envs.combinatorial_env import CombinatorialEnv
from algorithms.baselines import CombinatorialRandomAccess

n_seeds = 1
# n_agents_list = [4, 8, 12, 16]
loads_list = [1/7, 1/3.5, 1/1.75, 1/1.5, 1/1.25, 1]

n_agents = 6
n_channels = 16
channel_switch = np.array([0.8 for _ in range(n_channels)])
episode_length = 200

output_path = 'combinatorial_load/results/aloha_16_channels.p'

gf_scores_list = []
gf_jains_list = []
gf_channel_errors_list = []
gf_average_rewards_list = []


for seed in range(n_seeds):
    print(f"Seed {seed}")
    edf_scores_list_seed = []
    edf_jains_list_seed = []
    edf_channel_errors_list_seed = []
    edf_average_rewards_list_seed = []

    gf_scores_list_seed = []
    gf_jains_list_seed = []
    gf_channel_errors_list_seed = []
    gf_average_rewards_list_seed = []

    for l in loads_list:
        print(f"Load={l}")
        deadlines = np.array([7, 14] * (n_agents//2))
        lbdas = np.array([l] * (n_agents))
        period = np.array([1/l] * (n_agents))
        arrival_probs = np.array([0.4, 0.8] * (n_agents//2))
        offsets = np.zeros(n_agents)
        periodic_devices = np.array([0, 1])

        env = CombinatorialEnv(n_agents=n_agents,
                        n_channels=n_channels,
                        deadlines=deadlines,
                        lbdas=lbdas,
                        period=period,
                        arrival_probs=arrival_probs,
                        offsets=offsets,
                        episode_length=episode_length,
                        traffic_model='heterogeneous',
                        periodic_devices=periodic_devices,
                        channel_switch=channel_switch,
                        verbose=False)
        
        gf = CombinatorialRandomAccess(env)
        cv = gf.get_best_transmission_probs(100)
        gf.transmission_prob = gf.transmission_prob_list[np.argmax(cv)]
        score_gf, jains_gf, channel_error_gf, rewards_gf = gf.run(1000)

        gf_scores_list_seed.append(score_gf)
        gf_jains_list_seed.append(jains_gf)
        gf_channel_errors_list_seed.append(channel_error_gf)
        gf_average_rewards_list_seed.append(rewards_gf)

        print(f"URLLC score GF Access: {score_gf}")
        print(f"Jain's index GF Access: {jains_gf}")
        print(f"Channel errors GF Access: {channel_error_gf}")
        print(f"Reward per episode GF Access: {rewards_gf}\n")

    
    gf_scores_list.append(gf_scores_list_seed)
    gf_jains_list.append(gf_jains_list_seed)
    gf_channel_errors_list.append(gf_channel_errors_list_seed)
    gf_average_rewards_list.append(gf_average_rewards_list_seed)

ma_baselines_result = {
                       "gf_scores": gf_scores_list, 'gf_jains': gf_jains_list, "gf_channel_errors": gf_channel_errors_list, "gf_average_rewards": gf_average_rewards_list
                       }


pickle.dump(ma_baselines_result, open(output_path, 'wb'))
