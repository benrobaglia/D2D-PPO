import random
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim 
from collections import namedtuple, deque
import torch

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'state_next', 'done')
)


class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = deque(maxlen=buffer_limit)
        self.device = device
        self.buffer_limit = buffer_limit

    def add(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = s_lst[0].shape[0], s_lst[0].shape[1]
        return torch.stack(s_lst).view(batch_size, chunk_size, n_agents, obs_size).to(self.device), \
               torch.tensor(np.stack(a_lst), dtype=torch.long).view(batch_size, chunk_size, n_agents).to(self.device), \
               torch.tensor(np.stack(r_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents).to(self.device), \
               torch.stack(s_prime_lst).view(batch_size, chunk_size, n_agents, obs_size).to(self.device), \
               torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1).to(self.device)

    def reset(self):
        self.buffer = deque(maxlen=self.buffer_limit)

    def __len__(self):
        return len(self.buffer)


def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=100, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.GRU(n_inputs, hidden_size, 1)
        layers = []
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
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
        return self.layers(out)

class DQN:
    def __init__(
        self,
        env,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        adam_epsilon=1e-8,
        loss='huber'
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.adam_epsilon = adam_epsilon
        if callable(loss):
            self.loss = loss
        else:
            self.loss = {'huber':torch.functional.F.smooth_l1_loss, 'mse': torch.functional.F.mse_loss}[loss]
        
        self.env = env

        state_size = self.env.observation_space[0].shape[0]
        action_size = self.env.action_space[0].n

        self.network = RNN(state_size, action_size).to(self.device)
        self.target_network = RNN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)


    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

        with torch.no_grad():
            q_value_target = self.target_network(states_next).max(1, True)[0]
 
        td_target = rewards + (1 - dones) * self.gamma * q_value_target
        q_value = self.network(states).gather(1, actions)

        loss = self.loss(q_value, td_target, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0,1) >= self.epsilon:
            action = self.predict(state)
        else:
            action = np.random.randint(0, 2)
            # action = self.env.timestep % self.env.n
            
        return action 
        
    def update_epsilon(self, timestep, horizon_eps=1000):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (timestep / horizon_eps)
        self.epsilon = max(eps, self.final_exploration_rate)

    @torch.no_grad()
    def predict(self, state):
        action = self.network(state).argmax().item()
        return action

class iRDQN:
    def __init__(
        self,
        env,
        history_len = 5,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        adam_epsilon=1e-8,
        loss='huber',
        early_stopping=True
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history_len = history_len
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.adam_epsilon = adam_epsilon
        self.early_stopping = early_stopping
        if callable(loss):
            self.loss = loss
        else:
            self.loss = {'huber':torch.functional.F.smooth_l1_loss, 'mse': torch.functional.F.mse_loss}[loss]
        
        self.env = env
        self.agents = [DQN(env,
                        replay_start_size=replay_start_size,
                        replay_buffer_size=replay_buffer_size,
                        gamma=gamma,
                        update_target_frequency=update_target_frequency,
                        minibatch_size=minibatch_size,
                        learning_rate=learning_rate,
                        update_frequency=update_frequency,
                        initial_exploration_rate=initial_exploration_rate,
                        final_exploration_rate=final_exploration_rate,
                        adam_epsilon=adam_epsilon,
                        loss=loss) for _ in range(env.n_agents)]


    def train(self, n_episodes, early_stopping=True):
        
        received_list = []
        discarded_list = []
        test_list = []
        reward_list = []
        train_scores = []

        for ep in range(n_episodes):
            history = []
            score = 0
            is_training_ready = ep >= self.replay_start_size
            done = False
            state, _ = self.env.reset()
            state = torch.tensor(np.stack(state), dtype=torch.float).to(self.device)
            history.append(state)
    
            while not done:
                actions = []
                actions_binary = []
                history_tensor = torch.stack(history) # dim: (L, n, state_size)

                for i in range(self.env.n_agents):
                    a = self.agents[i]
                    obs = history_tensor[:, i, :]
                    action = a.act((obs.to(a.device)),
                                        is_training_ready=is_training_ready)
                    action_binary = np.zeros(self.env.n_channels)
                    action_binary[action] = 1
                    actions.append(action)
                    actions_binary.append(action_binary)
                    a.update_epsilon(ep)
                actions = np.array(actions)
                actions_binary = np.stack(actions_binary)

                state_next, _, rewards, done, _ = self.env.step(actions_binary)
                score += rewards.sum()
                
                state_next = torch.tensor(np.stack(state_next), dtype=torch.float).to(self.device)
                
                # Update history
                history.append(state_next)
                if len(history) > self.history_len:
                    del history[0]

                self.replay_buffer.add((state, actions, rewards, state_next, done))
                state = state_next

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets.sum())
            train_scores.append(1 - self.env.discarded_packets.sum() / self.env.received_packets.sum())

            if ep % 100 == 0:
                ts, tr = self.test(50)
                test_list.append(ts)
                reward_list.append(tr)
                print(f"Episode: {ep}, Running reward: {score}, Test score: {ts}, eps: {self.agents[0].epsilon}, {len(train_scores)}")
                # Early stopping: 
                if (early_stopping) & (ts == 1):
                    print(f"Early stopping at episode {ep}")
                    break
                
                
            if is_training_ready:
                if ep % self.update_frequency == 0:
                    transitions = self.replay_buffer.sample_chunk(self.minibatch_size, 
                                                                    self.history_len
                                                                    )
                    for i in range(self.env.n_agents):
                        a = self.agents[i]
                        tr = (transitions[0][:, :, i, :],
                                transitions[1][:, -1, i].unsqueeze(1),
                                transitions[2][:, -1, i].unsqueeze(1),
                                transitions[3][:, :, i, :],
                                transitions[4][:, -1, :]
                        )
                        loss = a.train_step(tr)
                        if ep % a.update_target_frequency == 0:
                            a.target_network.load_state_dict(a.network.state_dict())
        
        return train_scores, test_list, reward_list


    def test(self, n_episodes, verbose=False):
        
        received_list = []
        discarded_list = []
        reward_list = []

        for ep in range(n_episodes):
            history = []
            state, _ = self.env.reset()
            state = torch.tensor(np.stack(state), dtype=torch.float).to(self.device)
            history.append(state)

            done = False
            score = 0

            while not done:
                actions = []
                history_tensor = torch.stack(history) # dim: (L, n, state_size)
                for i,a in enumerate(self.agents):
                    obs = history_tensor[:, i, :]
                    action_idx = a.predict(obs.to(self.device))
                    action_agent = np.zeros(self.env.n_channels)
                    action_agent[action_idx] = 1
                    actions.append(action_agent)
                    
                state_next, _, reward, done, _ = self.env.step(np.stack(actions))
                state_next = torch.tensor(np.stack(state_next), dtype=torch.float).to(self.device)       

                # Update history
                history.append(state_next)
                if len(history) > self.history_len:
                    del history[0]

                score += np.mean(np.maximum(reward, 0))
                
                if verbose:
                    print(f"Timestep: {self.env.timestep}")
                    print(f"State: {state}")
                    print(f"History: {history}")
                    print(f"Next state: {state_next}")
                    print(f"Action: {actions}")
                    print(f"Reward: {reward}")
                    print(f"Number discarded: {self.env.discarded_packets.sum()}")

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets.sum())
            reward_list.append(score)
                                
        return 1 - np.sum(discarded_list) / np.sum(received_list), np.mean(reward_list)

