import numpy as np
import random


class RandomAccess:
    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose
    
    def act(self, buffers):
        n_packets = buffers.reshape((self.env.n_agents, self.env.deadlines.max())).sum(1)
        actions = np.random.choice(np.arange(0, self.env.n_channels+1), size=self.env.n_agents)
        actions[n_packets == 0] = 0
        return actions
    
    def run(self, n_episodes):
        number_of_discarded = []
        number_of_received = []
        rewards_list = []
        jains_index = []
        channel_score = []
        for _ in range(n_episodes):
            rewards_episode = []
            done = False
            _, (buffer_state, channel_state) = self.env.reset()

            while not done:

                action = self.act(buffer_state)
                _, next_state, reward, done, _ = self.env.step(action)
                buffer_state = next_state[0]
                rewards_episode.append(reward)

            rewards_list.append(np.sum(rewards_episode))
            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets.sum())
            jains_index.append(self.env.compute_jains())
            channel_score.append(self.env.compute_channel_score())
            
            
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")
            print(f"Channel score: {np.mean(channel_score)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received), np.mean(jains_index), np.mean(channel_score), np.mean(rewards_list)


class EarliestDeadlineFirstScheduler:
    def __init__(self, env, use_channel=False, verbose=False):
        self.env = env
        self.use_channel = use_channel
        self.verbose = verbose
        self.name = "EDF"

    def preprocess_state(self, state):
        output = []
        for row in state:
            packets = np.nonzero(row)[0]
            if len(packets) > 0:
                output.append(packets.min())
            else:
                output.append(-1)
        return np.array(output)
    
    def act(self, buffers):
        agg_state = self.preprocess_state(buffers)
        n_packets = (agg_state >= 0).sum()
        if n_packets > 0:
            has_a_packet = (agg_state + 1).nonzero()[0]
            sorted_idx = agg_state[has_a_packet].argmin()
            action_idx = has_a_packet[sorted_idx]
        else:
            action_idx = np.random.randint(self.env.n_agents)
        actions = np.zeros(self.env.n_agents)
        actions[action_idx] = 1.
        return actions

    def run(self, n_episodes):
        number_of_discarded = []
        number_of_received = []
        rewards_list = []
        jains_index = []
        number_channel_losses = []
        for _ in range(n_episodes):
            rewards_episode = []
            done = False
            _, (buffer_state, channel_state) = self.env.reset()

            while not done:
                # Filter the good/bad channels
                if self.use_channel:
                    channel_state = channel_state > 0.5
                    bad_channels = (channel_state == 0)
                    buffer_state[bad_channels] = 0

                action = self.act(buffer_state)
                _, next_state, reward, done, _ = self.env.step(action)
                buffer_state = next_state[0]
                rewards_episode.append(reward)

            rewards_list.append(np.sum(rewards_episode))
            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets.sum())
            jains_index.append(self.env.compute_jains())
            number_channel_losses.append(self.env.channel_errors)
        
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")
            print(f"Number of channel_losses: {np.sum(number_channel_losses)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received), np.mean(jains_index), np.sum(number_channel_losses), np.mean(rewards_list)
    
class GFAccess:
    def __init__(self, env, transmission_prob=0.5, transmission_prob_list=[0.1, 0.2 ,0.3 ,0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1], use_channel=False, verbose=False):
        self.env = env
        self.transmission_prob = transmission_prob
        self.transmission_prob_list = transmission_prob_list
        self.use_channel = use_channel
        self.verbose = verbose
    
    def act(self, buffers):
        n_packets = buffers.sum(1)
        actions = np.random.binomial(1, p=self.transmission_prob, size=self.env.n_agents)
        actions[n_packets == 0] = 0
        return actions

    def get_best_transmission_probs(self, n_episodes):
        cv = []
        for tp in self.transmission_prob_list:
            self.transmission_prob = tp
            score, _, _, _ = self.run(n_episodes)
            cv.append(np.mean(score))
        return cv
    
    def run(self, n_episodes):
        number_of_discarded = []
        number_of_received = []
        rewards_list = []
        jains_index = []
        number_channel_losses = []
        for _ in range(n_episodes):
            rewards_episode = []
            done = False
            _, state = self.env.reset()

            while not done:
                # Filter the good/bad channels
                if self.use_channel:
                    channel_state = state[1] > 0.5
                    bad_channels = (channel_state == 0)
                    buffer_state[bad_channels] = 0

                action = self.act(buffer_state)
                _, next_state, reward, done, _ = self.env.step(action)
                buffer_state = next_state[0]
                rewards_episode.append(reward)

            rewards_list.append(np.sum(rewards_episode))
            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets.sum())
            jains_index.append(self.env.compute_jains())
            number_channel_losses.append(self.env.channel_errors)
        
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")
            print(f"Number of channel_losses: {np.sum(number_channel_losses)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received), np.mean(jains_index), np.sum(number_channel_losses), np.mean(rewards_list)


class CombinatorialRandomAccess:
    def __init__(self, env, transmission_prob=0.5, transmission_prob_list=None, verbose=False):
        self.env = env
        self.transmission_prob = transmission_prob
        if transmission_prob_list is None:
            self.transmission_prob_list = np.arange(0, 1, 0.1)
        else:
            self.transmission_prob_list = transmission_prob_list
        self.verbose = verbose
    
    def act(self, buffers):
        actions = np.random.binomial(1, self.transmission_prob, (self.env.n_agents, self.env.n_channels))
        return actions

    def get_best_transmission_probs(self, n_episodes):
        cv = []
        for tp in self.transmission_prob_list:
            self.transmission_prob = tp
            score, _, _, _ = self.run(n_episodes)
            cv.append(np.mean(score))
        return cv

    def run(self, n_episodes):
        number_of_discarded = []
        number_of_received = []
        rewards_list = []
        jains_index = []
        channel_score = []
        for _ in range(n_episodes):
            rewards_episode = []
            done = False
            _, state = self.env.reset()

            while not done:

                action = self.act(state[0])
                _, next_state, reward, done, _ = self.env.step(action)
                buffer_state = next_state[0]
                rewards_episode.append(reward)

            rewards_list.append(np.sum(rewards_episode))
            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets.sum())
            jains_index.append(self.env.compute_jains())
            channel_score.append(self.env.compute_channel_score())
            
            
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")
            print(f"Channel score: {np.mean(channel_score)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received), np.mean(jains_index), np.mean(channel_score), np.mean(rewards_list)
