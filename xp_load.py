import numpy as np
import torch
import pickle
from envs.combinatorial_env import CombinatorialEnv
from algorithms.d2d_ppo import D2DPPO
from algorithms.irdqn import iRDQN
from algorithms.ippo import iPPO
import os
import random

# Fix random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

xp_name = 'combinatorial_load'
output_path = f'{xp_name}/results/idrqn_16_channels.p'

if xp_name not in os.listdir():
    print(f"Creating directories for experiment...")
    os.mkdir(f'{xp_name}')
    os.mkdir(f'{xp_name}/results')

# Initializing experiment

print("Device: ", torch.device('cuda' if torch.cuda.is_available() else "cpu"))
print(f"Launching {xp_name} experiment...")
print(f"output_path: {output_path}")

n_seeds = 1
setup = pickle.load(open("combinatorial_load/setup.p", "rb"))

print(setup)

ppo_scores_list = []
ppo_jains_list = []
ppo_channel_errors_list = []
ppo_average_rewards_list = []
training_list = []

# Launch experiment

n_agents = setup['n_agents']

for seed in range(n_seeds):
    print(f"Seed {seed}")
    ppo_scores_list_seed = []
    ppo_jains_list_seed = []
    ppo_channel_errors_list_seed = []
    ppo_average_rewards_list_seed = []
    training_list_seed = []

    for load in setup['loads_list']:
        print(f"load= {load}")
        # Managing directories for models
        model_folder = f"models_ippo{setup['n_channels']}_seed_{seed}_load_{load}"
        if model_folder not in os.listdir(xp_name):
            os.mkdir(f"{xp_name}/{model_folder}")

        lbdas = np.array([load] * (n_agents))
        period = np.array([int(1/load)] * (n_agents))

        env = CombinatorialEnv(n_agents=n_agents,
                        n_channels=setup['n_channels'],
                        deadlines=setup['deadlines'],
                        lbdas=lbdas,
                        period=period,
                        arrival_probs=setup['arrival_probs'],
                        offsets=setup['offsets'],
                        episode_length=setup["episode_length"],
                        traffic_model='heterogeneous',
                        homogeneous_size=True,
                        periodic_devices=setup['periodic_devices'],
                        channel_switch=setup['channel_switch'],
                        verbose=False)
        
        # MCAPPO
        # ppo = D2DPPO(env, 
        #                 hidden_size=64, 
        #                 gamma=0.6,
        #                 policy_lr=3e-4,
        #                 value_lr=1e-3,
        #                 device=None,
        #                 useRNN=True,
        #                 save_path=f"{xp_name}/{model_folder}",
        #                 combinatorial=True,
        #                 history_len=n_agents,
        #                 early_stopping=True
        #                 )
        

        # MCA-iPPO
        # ppo = iPPO(env, 
        #             hidden_size=64, 
        #             gamma=0.4,
        #             policy_lr=3e-4,
        #             value_lr=1e-3,
        #             device=None,
        #             useRNN=True,
        #             save_path=f"{xp_name}/{model_folder}",
        #             combinatorial=True,
        #             history_len=n_agents,
        #             early_stopping=True
        #             )
        
        # res = ppo.train(num_iter=2000, n_epoch=5, num_episodes=10, test_freq=100)    
        # ppo.load(f"{xp_name}/{model_folder}")
        # score_ppo, jains_ppo, channel_error_ppo, rewards_ppo = ppo.test(1000)


        #iDRQN 
        idqn = iRDQN(
                    env,
                    history_len = n_agents,
                    replay_start_size=100,
                    replay_buffer_size=100000,
                    gamma=0.4,
                    update_target_frequency=100,
                    minibatch_size=64,
                    learning_rate=1e-4,
                    update_frequency=1,
                    initial_exploration_rate=1,
                    final_exploration_rate=0.1,
                    adam_epsilon=1e-8,
                    loss='huber'
                )

        res = idqn.train(20000)
        score_ppo, jains_ppo = idqn.test(500)
        channel_error_ppo = ""
        rewards_ppo = ""


        print(f"URLLC score ppo: {score_ppo}")
        print(f"Jain's index ppo: {jains_ppo}")
        print(f"Channel errors ppo: {channel_error_ppo}")
        print(f"Reward per episode ppo: {rewards_ppo}\n")

        training_list_seed.append(res)

        ppo_scores_list_seed.append(score_ppo)
        ppo_jains_list_seed.append(jains_ppo)
        ppo_channel_errors_list_seed.append(channel_error_ppo)
        ppo_average_rewards_list_seed.append(rewards_ppo)
    
    training_list.append(training_list_seed)
    ppo_scores_list.append(np.array(ppo_scores_list_seed))
    ppo_jains_list.append(np.array(ppo_jains_list_seed))
    ppo_channel_errors_list.append(np.array(ppo_channel_errors_list_seed))
    ppo_average_rewards_list.append(np.array(ppo_average_rewards_list_seed))



ppo_result = {"scores": ppo_scores_list,
                "jains": ppo_jains_list, 
                "channel_errors": ppo_channel_errors_list, 
                "average_rewards": ppo_average_rewards_list,
                "training": training_list
               }


pickle.dump(ppo_result, open(output_path, 'wb'))