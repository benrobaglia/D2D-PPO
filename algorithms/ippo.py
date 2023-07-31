import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Policy(nn.Module):
    def __init__(self, num_inputs, n_actions, hidden_size=100):
        super(Policy, self).__init__()
        self.n_actions = n_actions
        num_outputs = n_actions

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        
        init_weights(self.linear1, 2)
        init_weights(self.linear2, 2)

    def forward(self, inputs):
        if len(inputs.shape) == 1:
            x = inputs.unsqueeze(0)
        else:
            x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=1)

class Value(nn.Module):
    def __init__(self, num_inputs, hidden_size=100):
        super(Value, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        
        init_weights(self.linear1, 2)
        init_weights(self.linear2, 2)
    
    def forward(self, inputs):
        if len(inputs.shape) == 1:
            x = inputs.unsqueeze(0)
        else:
            x = inputs
        x = F.relu(self.linear1(x))
        return self.linear2(x)

def compute_gae(rewards, dones, values, gamma, lbda=0.95):
    gae = 0
    adv = [rewards[-1] - values[-1]]
    for step in reversed(range(len(rewards)-1)):
        delta = rewards[step] + gamma * values[step + 1] * (1-dones[step]) - values[step]
        gae = delta + gamma * lbda * (1-dones[step]) * gae
        adv.insert(0, gae + values[step])
    adv = np.array(adv)
    if (adv.std(0) > 0).all():
        adv = (adv - adv.mean(0)) / adv.std(0)
    return torch.tensor(adv, dtype=torch.float)

def discount_rewards(rewards, gamma, dones, normalize=True):
    returns = []
    R = 0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + R * gamma * (1 - dones[i])
        returns.insert(0, R)

    returns = torch.tensor(np.array(returns), dtype=torch.float)
    
    if normalize:
        if (returns.std(0) > 0).all():
            returns = (returns - returns.mean(0)) / returns.std(0)
    return returns


def create_rollouts(env, agents, num_episodes=4):
    # Simulate the rolling policy for num_episodes
    device = agents[0].device
    states = []
    actions = []
    log_probs = []
    # entropies = []
    rewards = []
    scores = []
    values = []
    dones = []
    
    for _ in range(num_episodes):
        done = False
        rewards_episode = []
        state = torch.tensor(env.reset(), dtype=torch.float).to(device)
        while not done:
            states.append(state)
            action_agent = []
            log_prob_agent = []
            # entropy_agent = []
            value_agent = []
            for i, agent in enumerate(agents):
                action, log_prob, entropy = agent.select_action(state[i], train=True)
                action_agent.append(action)
                log_prob_agent.append(log_prob)
                # entropy_agent.appent(entropy)
                value = agent.value_network(state[i]).item()
                value_agent.append(value)
            
            next_state, reward, done = env.step(action_agent)
            
            dones.append(done)
            actions.append(action_agent)
            log_probs.append(log_prob_agent)
            # entropies.append(entropy_agent)
            values.append(value_agent)
            rewards_episode.append(reward)
            rewards.append(reward)
            state = torch.as_tensor(next_state, dtype=torch.float).to(device)

        n_received = env.received_packets.sum() - (env.current_state).sum()
        if n_received > 0:
            score = env.successful_transmissions / n_received        
        else:
            score = 1.
        
        scores.append(score)

    values = np.array(values)
    rewards = np.stack(rewards)
    advantages = compute_gae(rewards, dones, values, agents[0].gamma, 0.97)
    returns = discount_rewards(rewards, agents[0].gamma, dones, True)
    log_probs = torch.tensor(log_probs)
    # entropies = torch.stack(entropies)

    return torch.stack(states), np.array(actions), log_probs, returns, values, advantages, scores, dones

class PPO:
    def __init__(self, num_inputs, n_actions, hidden_size=128, gamma=0.99, policy_lr=1e-3, value_lr=1e-3, device='cpu', early_stopping=True):
        self.early_stopping = early_stopping
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.policy_network = Policy(num_inputs, n_actions, hidden_size)
        self.value_network = Value(num_inputs, hidden_size)
        self.gamma = gamma
        self.policy_network = self.policy_network.to(self.device)
        self.value_network = self.value_network.to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=value_lr)

    def select_action(self, state, train=True):
        probs = self.policy_network(state.to(self.device))    
        dist = Categorical(probs)
        if train:
            action = dist.sample()
        else:
            action = probs.argmax(dim=1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy

    def evaluate(self, states, actions):
        # States : batch * state_size
        # Actions: list
        probs = self.policy_network(states.to(self.device).squeeze())
        dist = Categorical(probs)
        log_probs = dist.log_prob(torch.tensor(actions).to(self.device))
        return log_probs, dist.entropy()
    
    def train_step(self, states, actions, log_probs_old, returns, advantages, cliprange=0.1, beta=0.01):
        # Update policy
        log_probs, entropy = self.evaluate(states, actions)
        entropy = entropy.mean()

        ratio = torch.exp(log_probs - log_probs_old.to(self.device))
        surr1 = ratio * advantages.to(self.device)
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages.to(self.device)
        policy_loss = - torch.min(surr1, surr2).mean() - beta * entropy
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 40)
        self.policy_optimizer.step()
        
        # Update value
        mse = torch.functional.F.mse_loss
        value = self.value_network(states.to(self.device)).squeeze()
        value_loss = mse(value, returns.to(self.device), reduction='mean')
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 40)
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()


def train_ppo(env, agents, num_iter, num_episodes=4):
    scores_episode = []
    score_test_list = []
    
    for iter in range(num_iter):
        states, actions, log_probs_old, returns, _, advantages, scores, _ = create_rollouts(env, agents, num_episodes)

        # Shape: (len_trajectory * num agents)
        scores_episode += scores

        if states.sum() > 0:
            for i in range(env.n):
                loss = agents[i].train_step(states[:, i, :], actions[:,i], log_probs_old[:, i], returns[:,i], advantages[:, i])
        
        if iter % 100 == 0:
            score_test = test_ppo(env, agents, 100)
            score_test_list.append(score_test)
            if (score_test == 1) & (agents[0].early_stopping == True):
                break
            print(f"Episode: {iter}, mean score rollout: {np.mean(scores)} Score test: {score_test}")


#                 print(loss)
#             print(states)
#             print(actions)
#             print(rewards)
#             print("\n")
            
                
    return scores_episode, score_test_list

	

class iPPO:
    def __init__(self,
                env,
                hidden_size=128, 
                gamma=0.99, 
                policy_lr=1e-3,
                value_lr=1e-3,
                device=None,
                early_stopping=True):
        
        self.env = env
        self.n_agents = env.n_agents
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.policy_lr=policy_lr, 
        self.value_lr=value_lr,
        self.early_stopping = early_stopping
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.agents = [PPO(num_inputs=self.env.observation_space[k].shape[0],
                           n_actions=self.env.action_space[k].n,
                           hidden_size=hidden_size,
                           gamma=gamma,
                           policy_lr=policy_lr,
                           value_lr=value_lr,
                           early_stopping=early_stopping) for k in range(self.n_agents)]
        
    def create_rollouts(self, num_episodes=4):
        # Simulate the rolling policy for num_episodes
        buffer_states = []
        channel_states = []
        observations = {f"{i}": [] for i in range(self.n_agents)}
        actions = []
        log_probs = []
        # entropies = []
        rewards = []
        scores = []
        values = []
        dones = []
        
        for _ in range(num_episodes):
            done = False
            rewards_episode = []
            obs, (buffer_state, channel_state) = self.env.reset()
            while not done:
                buffer_states.append(buffer_state)
                channel_states.append(channel_state)
                action_agent = []
                log_prob_agent = []
                # entropy_agent = []
                value_agent = []
                for i, agent in enumerate(self.agents):
                    obs_agent = torch.tensor(obs[i], dtype=torch.float).to(self.device)
                    observations[str(i)].append(obs_agent)
                    action, log_prob, entropy = agent.select_action(obs_agent, train=True)
                    action_agent.append(action)
                    log_prob_agent.append(log_prob)
                    # entropy_agent.appent(entropy)
                    value = agent.value_network(obs_agent).item()
                    value_agent.append(value)
                
                next_obs, next_state, reward, done, _ = self.env.step(np.array(action_agent))
                
                dones.append(done)
                actions.append(action_agent)
                log_probs.append(log_prob_agent)
                # entropies.append(entropy_agent)
                values.append(value_agent)
                rewards_episode.append(reward)
                rewards.append(reward)
                state = next_state
                obs = next_obs

            score = 1 - self.env.discarded_packets.sum() / self.env.received_packets.sum()
            scores.append(score)

        values = np.array(values)
        rewards = np.stack(rewards)
        advantages = compute_gae(rewards, dones, values, self.gamma, 0.97)
        returns = discount_rewards(rewards, self.gamma, dones, True)
        log_probs = torch.tensor(log_probs)
        # entropies = torch.stack(entropies)
        obs_tensor = [torch.stack(observations[str(i)]) for i in range(self.n_agents)]

        return obs_tensor, np.array(actions), log_probs, returns, values, advantages, scores, dones

    def test(self, num_episodes):
        scores_episode = []
        average_rewards = []
        number_of_received = []
        number_of_discarded = []
        jains_index = []
        number_channel_losses = []

        for i_episode in range(num_episodes):
            running_rewards = []
            obs, _ = self.env.reset()
            done = False
            while not done:
                actions = []
                for i, agent in enumerate(self.agents):
                    obs_agent = torch.tensor(obs[i], dtype=torch.float).to(self.device)
                    action, _, _ = agent.select_action(obs_agent, train=True)
                    actions.append(action)

                next_obs, _, reward, done, _ = self.env.step(actions)
                running_rewards.append(reward.mean())
                obs = next_obs
            
            average_rewards.append(np.sum(running_rewards))
            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets.sum())
            jains_index.append(self.env.compute_jains())
            number_channel_losses.append(self.env.channel_errors)
            scores_episode.append(1-self.env.discarded_packets.sum() / self.env.received_packets.sum())
                    
        return np.mean(scores_episode), np.mean(jains_index), np.sum(number_channel_losses), np.mean(average_rewards)

    def train(self, num_iter, num_episodes=4, test_freq=100):
        scores_episode = []
        score_test_list = []
        
        for iter in range(num_iter):
            obs, actions, log_probs_old, returns, _, advantages, scores, _ = self.create_rollouts(num_episodes)

            # Shape: (len_trajectory * num agents)
            scores_episode += scores

            for i in range(self.n_agents):
                loss = self.agents[i].train_step(obs[i], actions[:,i], log_probs_old[:, i], returns[:,i], advantages[:, i])
            
            if iter % test_freq == 0:
                score_test = self.test(50)
                score_test_list.append(score_test)
                if (score_test[0] == 1) & (self.early_stopping):
                    break
                print(f"Episode: {iter}, mean score rollout: {np.mean(scores)} Score test: {score_test}")
                    
        return scores_episode, score_test_list
