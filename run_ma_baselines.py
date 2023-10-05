import numpy as np
import pickle
from envs.combinatorial_env import CombinatorialEnv
from algorithms.baselines import CombinatorialRandomAccess

n_seeds = 1

output_path = 'combinatorial_load/results/aloha_16_channels.p'

# Run channel switch first time
# channel_switch = np.random.choice([0.2 ,0.4, 0.6, 0.8], size=(n_agents, n_channels))
# pickle.dump(channel_switch, open("combinatorial_load/channel_switch.p", 'wb'))

# Load channel switch
channel_switch = pickle.load(open("combinatorial_load/channel_switch.p", 'rb'))

print(f"Channel Switch: {channel_switch}")
print(f"Channel Switch Mean: {channel_switch.mean()}")

# Create setup
setup = {"n_agents": 6,
         "n_channels": 16,
         "episode_length": 200,
         "loads_list": [1/3, 1/2, 1/1.5, 1/1.25, 1],
         "deadlines": np.array([7, 14] * 3),
         "arrival_probs": np.array([0.2, 0.4, 0.8, 1, 1, 1]),
         "offsets": np.zeros(6),
         "periodic_devices": np.array([0, 1, 2]),
         "channel_switch": channel_switch
         }

pickle.dump(setup, open("combinatorial_load/setup.p", 'wb'))

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

    n_agents = setup['n_agents']

    for l in setup['loads_list']:
        print(f"Load={l}")
        lbdas = np.array([l] * (n_agents))
        period = np.array([int(1/l)] * (n_agents))

        env = CombinatorialEnv(n_agents=n_agents,
                        n_channels=setup['n_channels'],
                        deadlines=setup['deadlines'],
                        lbdas=lbdas,
                        period=period,
                        arrival_probs=setup['arrival_probs'],
                        offsets=setup['offsets'],
                        episode_length=setup["episode_length"],
                        traffic_model='heterogeneous',
                        periodic_devices=setup['periodic_devices'],
                        channel_switch=setup['channel_switch'],
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
