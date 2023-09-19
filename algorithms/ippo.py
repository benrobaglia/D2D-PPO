import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=100, combinatorial=False, use_activation=True, **kwargs):

        super().__init__()
        self.hidden_size = hidden_size
        self.combinatorial = combinatorial
        self.use_activation = use_activation
        self.lstm = nn.GRU(n_inputs, hidden_size, 1)
        layers = []
        # layers.append(nn.Linear(hidden_size, hidden_size))
        # layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, n_outputs))
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(lambda x: init_weights(x, 3))
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, obs):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0) # Add batch dimension
            
        batch_size = obs.size(0)
        self.hidden = self.init_hidden(batch_size=batch_size)
        lstm_out, self.hidden = self.lstm(obs.permute(1, 0, 2))
        out = lstm_out[-1]
        out = self.layers(out)
        if self.use_activation:
            if not self.combinatorial:
                out = F.softmax(out, dim=1)
            else:
                out = torch.sigmoid(out)
        return out


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



class PPO:
    def __init__(self,
                num_inputs,
                n_actions, 
                hidden_size=128, 
                gamma=0.99, 
                policy_lr=1e-3, 
                value_lr=1e-3, 
                useRNN=False,
                combinatorial=False,
                device='cpu', 
                history_len = 5,
                early_stopping=True):
        
        self.history_len = history_len
        self.useRNN = useRNN
        self.combinatorial = combinatorial
        self.early_stopping = early_stopping
        self.device = torch.device(device)
        self.n_actions = n_actions

        if not self.useRNN:
            self.policy_network = Policy(num_inputs, n_actions, hidden_size)
            self.value_network = Value(num_inputs, hidden_size)
        else:
            self.policy_network = RNN(num_inputs, n_actions, hidden_size, combinatorial=combinatorial)
            self.value_network = RNN(num_inputs, 1, hidden_size, combinatorial=False, use_activation=False)

        self.gamma = gamma
        self.policy_network = self.policy_network.to(self.device)
        self.value_network = self.value_network.to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=value_lr)

    def select_action(self, state, train=True):
        probs = self.policy_network(state.to(self.device))    
        if self.combinatorial:
            dist = Bernoulli(probs)
            if train:
                action = dist.sample().squeeze()
            else:
                action = dist.probs.squeeze() > 0.5
                action = action * 1.
            log_prob = dist.log_prob(action).mean(-1)
            entropy = dist.entropy().mean(-1)
                
        else:
            dist = Categorical(probs)
            if train:
                action = dist.sample()
            else:
                action = probs.argmax(dim=1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action.cpu().detach().numpy(), log_prob, entropy

    def evaluate(self, states, actions):
        # States : batch * state_size
        # Actions: list
        probs = self.policy_network(states.to(self.device).squeeze())
        if self.combinatorial:
            dist = Bernoulli(probs)
            log_probs = dist.log_prob(torch.tensor(actions).to(self.device)).mean(-1)
            dist_entropy = dist.entropy().mean(-1)
        else:
            dist = Categorical(probs)
            log_probs = dist.log_prob(torch.tensor(actions).to(self.device))
            dist_entropy = dist.entropy()

        return log_probs, dist_entropy


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
        self.policy_optimizer.step()
        
        # Update value
        mse = torch.functional.F.mse_loss
        value = self.value_network(states.to(self.device)).squeeze()
        value_loss = mse(value, returns.to(self.device), reduction='mean')
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()


	

class iPPO:
    def __init__(self,
                env,
                hidden_size=128, 
                gamma=0.99, 
                policy_lr=1e-3,
                value_lr=1e-3,
                device=None,
                useRNN=False,
                save_path=None,
                combinatorial=False,
                history_len=10,
                early_stopping=True):
        
        self.env = env
        self.history_len = history_len
        self.n_agents = env.n_agents
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.policy_lr=policy_lr 
        self.value_lr=value_lr
        self.early_stopping = early_stopping
        self.useRNN = useRNN
        self.combinatorial = combinatorial
        
        self.save_path = save_path

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
                           useRNN=useRNN,
                           combinatorial=combinatorial,
                           history_len=history_len,
                           device=self.device,
                           early_stopping=early_stopping) for k in range(self.n_agents)]
        
    def save(self, checkpoint_path):
        for i, agent in enumerate(self.agents):
            torch.save(agent.policy_network.state_dict(), f"{checkpoint_path}/agent_{i}.pth")
        print("Models saved!")

    def load(self, checkpoint_path):
        for i, agent in enumerate(self.agents):
            agent.policy_network.load_state_dict(torch.load(f"{checkpoint_path}/agent_{i}.pth"))
        print("Models loaded!")


    def create_rollouts(self, num_episodes=4):
        # Simulate the rolling policy for num_episodes
        states = []
        observations = {f"{i}": [] for i in range(self.n_agents)}
        actions_list = []
        log_probs = []
        # entropies = []
        rewards = []
        scores = []
        values = []
        dones = []
        
        for e in range(num_episodes):
            done = False
            rewards_episode = []
            obs, state = self.env.reset()
            while not done:
                states.append(state)
                action_agent = []
                log_prob_agent = []
                # entropy_agent = []
                value_agent = []
                for i, agent in enumerate(self.agents):
                    obs_agent = torch.tensor(obs[i], dtype=torch.float).to(self.device)
                    observations[str(i)].append(obs_agent)
                    if self.useRNN:
                        history_tensor = torch.stack(observations[str(i)][e*self.env.episode_length:][-self.history_len:]) # shape: (history_len, input_len)
                        action, log_prob, entropy = agent.select_action(history_tensor, train=True)
                        value = agent.value_network(history_tensor).item()
                    else:
                        action, log_prob, entropy = agent.select_action(obs_agent, train=True)
                        value = agent.value_network(obs_agent).item()

                    action_agent.append(action)
                    log_prob_agent.append(log_prob)
                    # entropy_agent.appent(entropy)
                    value_agent.append(value)
                
                if self.combinatorial:
                    actions = np.stack(action_agent)
                else:
                    actions = np.array(action_agent)

                next_obs, next_state, reward, done, _ = self.env.step(actions)
                
                dones.append(done)
                actions_list.append(actions)
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

        return obs_tensor, np.stack(actions_list), log_probs, returns, values, advantages, scores, dones

    def test(self, num_episodes):
        scores_episode = []
        average_rewards = []
        number_of_received = []
        number_of_discarded = []
        jains_index = []
        number_channel_losses = []
        observations = {f"{i}": [] for i in range(self.n_agents)}

        for e in range(num_episodes):
            running_rewards = []
            obs, _ = self.env.reset()
            done = False
            while not done:
                actions = []
                for i, agent in enumerate(self.agents):
                    obs_agent = torch.tensor(obs[i], dtype=torch.float).to(self.device)
                    observations[str(i)].append(obs_agent)

                    if self.useRNN:
                        history_tensor = torch.stack(observations[str(i)][e*self.env.episode_length:][-self.history_len:])
                        action, _, _ = agent.select_action(history_tensor, train=False)
                    else:
                        action, _, _ = agent.select_action(obs_agent, train=False)

                    actions.append(action)

                if self.combinatorial:
                    actions = np.stack(actions)
                else:
                    actions = np.array(actions)

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

    def preprocess_input_for_rnn(self, obs_agent):
        # obs_agent is the tensor of size (num_episodes * episode_length, input_size)
        # obs_agent_rnn is a tensor of size (num_episodes * episode_length, history_len, input_size)
        obs_agent_rnn = []
        for i in range(obs_agent.size(0)):
            idx = i % self.env.episode_length
            if idx < self.history_len:
                x = obs_agent[i-idx:i+1]
                pad_len = self.history_len - (idx + 1)
                x = torch.cat([torch.zeros((pad_len, obs_agent.size(1))).to(self.device), x]) # We pad with 0s.
            else:
                x = obs_agent[i+1-self.history_len:i+1]
            obs_agent_rnn.append(x)
        return torch.stack(obs_agent_rnn)


    def train(self, num_iter, n_epoch=4, num_episodes=4, test_freq=100):
        scores_episode = []
        score_test_list = []
        policy_loss_list = []
        value_loss_list = []

        for iter in range(num_iter):
            obs, actions, log_probs_old, returns, _, advantages, scores, _ = self.create_rollouts(num_episodes)

            # Shape: (len_trajectory * num agents)
            scores_episode += scores

            for epoch in range(n_epoch):
                for i in range(self.n_agents):
                    if self.useRNN:
                        obs_agent_rnn = self.preprocess_input_for_rnn(obs[i])
                        loss = self.agents[i].train_step(obs_agent_rnn, actions[:,i], log_probs_old[:, i], returns[:,i], advantages[:, i])
                    else:
                        loss = self.agents[i].train_step(obs[i], actions[:,i], log_probs_old[:, i], returns[:,i], advantages[:, i])
                policy_loss_list.append(loss[0])
                value_loss_list.append(loss[1])

                if iter % test_freq == 0:
                    score_test, jains, channel_loss, avg_rewards = self.test(50)
                    score_test_list.append(score_test)
                    print(f"Iteration: {iter}, Epoch: {epoch}, score rollout: {np.mean(scores)} Score test: {(score_test, jains, channel_loss, avg_rewards)}")
                    
                    if np.max(score_test_list) == score_test:
                        if self.save_path is not None:
                            self.save(self.save_path)

                    if (score_test == 1) & (self.early_stopping):
                        return scores_episode, score_test_list, policy_loss_list, value_loss_list
                    
                    
        return scores_episode, score_test_list, policy_loss_list, value_loss_list
