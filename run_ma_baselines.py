import numpy as np
import pickle
from envs.env import D2DEnv
from algorithms.baselines import EarliestDeadlineFirstScheduler, GFAccess

n_seeds = 5
n_agents_list = [2, 4, 6, 8, 10]

edf_scores_list = []
edf_jains_list = []
edf_channel_errors_list = []
edf_average_rewards_list = []

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

    for k in n_agents_list:
        print(f"k={k}")
        deadlines = np.array([7] * k)
        lbdas = np.array([1/14] * k)
        period = None
        arrival_probs = None
        offsets = None
        neighbourhoods = None

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
        
        edf = EarliestDeadlineFirstScheduler(env, use_channel=False, verbose=True)
        score_edf, jains_edf, channel_error_edf, rewards_edf = edf.run(500)

        edf_scores_list_seed.append(score_edf)
        edf_jains_list_seed.append(jains_edf)
        edf_channel_errors_list_seed.append(channel_error_edf)
        edf_average_rewards_list_seed.append(rewards_edf)
    
        print(f"URLLC score EDF: {score_edf}")
        print(f"Jain's index EDF: {jains_edf}")
        print(f"Channel errors EDF: {channel_error_edf}")
        print(f"Reward per episode EDF: {rewards_edf}\n")

        gf = GFAccess(env, use_channel=False)
        cv = gf.get_best_transmission_probs(100)
        gf.transmission_prob = gf.transmission_prob_list[np.argmax(cv)]
        score_gf, jains_gf, channel_error_gf, rewards_gf = gf.run(500)

        gf_scores_list_seed.append(score_gf)
        gf_jains_list_seed.append(jains_gf)
        gf_channel_errors_list_seed.append(channel_error_gf)
        gf_average_rewards_list_seed.append(rewards_gf)

        print(f"URLLC score GF Access: {score_gf}")
        print(f"Jain's index GF Access: {jains_gf}")
        print(f"Channel errors GF Access: {channel_error_gf}")
        print(f"Reward per episode GF Access: {rewards_gf}\n")

    
    edf_scores_list.append(edf_scores_list_seed)
    edf_jains_list.append(edf_jains_list_seed)
    edf_channel_errors_list.append(edf_channel_errors_list_seed)
    edf_average_rewards_list.append(edf_average_rewards_list_seed)

    gf_scores_list.append(gf_scores_list_seed)
    gf_jains_list.append(gf_jains_list_seed)
    gf_channel_errors_list.append(gf_channel_errors_list_seed)
    gf_average_rewards_list.append(gf_average_rewards_list_seed)

ma_baselines_result = {"edf_scores": edf_scores_list, "edf_jains": edf_jains_list, "edf_channel_errors": edf_channel_errors_list, "edf_average_rewards": edf_average_rewards_list,
                       "gf_scores": gf_scores_list, 'gf_jains': gf_jains_list, "gf_channel_errors": gf_channel_errors_list, "gf_average_rewards": gf_average_rewards_list,
                       "xp_params": {'n_agents_list': n_agents_list, 'deadlines': 7, 'interarrivals':14}
                       }


pickle.dump(ma_baselines_result, open('results/ma_baselines.p', 'wb'))