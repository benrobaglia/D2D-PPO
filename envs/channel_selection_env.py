import numpy as np
from gym import spaces

class ChannelSelectionEnv:
    def __init__(self,
                n_agents,
                n_channels,
                deadlines,
                lbdas,
                period=5,
                arrival_probs=None,
                offsets=None,
                episode_length=100,
                traffic_model='aperiodic',
                periodic_devices=[],
                reward_type=0,
                channel_switch=None,
                verbose=False): 
        
        self.verbose = verbose
        self.n_agents = n_agents
        self.n_channels = n_channels
        self.lbdas = lbdas
        self.period = period
        self.deadlines = deadlines
        self.arrival_probs = arrival_probs
        self.offsets = offsets
        self.episode_length = episode_length
        self.traffic_model = traffic_model
        self.reward_type = reward_type
        self.periodic_devices = periodic_devices
        self.aperiodic_devices = [i for i in range(self.n_agents) if i not in periodic_devices]

        # Channel
        if channel_switch is None:
            self.channel_switch = np.zeros(self.n_agents)
        else:
            self.channel_switch = channel_switch

        # Observation of device k: buffer of size the deadline + channel state + last feedback
        self.observation_space = spaces.Tuple([spaces.Box(low=-float('inf'), high=float('inf'),
                                                           shape=(self.deadlines[k] + self.n_channels+1,)) for k in range(self.n_agents)])
        self.action_space = spaces.Tuple([spaces.Discrete(self.n_channels+1) for _ in range(self.n_agents)])

        self.state_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                                        shape=(self.deadlines.sum() + self.n_channels + 1,))


    def reset(self):
        self.current_buffers = np.zeros((self.n_agents, np.max(self.deadlines)))  # The buffer status of all devices.
        # Set arrival times
        self.arrival_times = [[] for _ in range(self.n_agents)]
        
        if self.traffic_model == 'aperiodic':
            for i in range(self.n_agents):
                self.current_buffers[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
        
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.offsets == 0)[0]
            for ao in active_offsets:
                self.current_buffers[int(ao), self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
        
        elif self.traffic_model == 'heterogeneous':
            assert (self.periodic_devices != [] and self.aperiodic_devices != []), "periodic_devices and aperiodic_devices must be non empty"

            for i in self.aperiodic_devices:
                self.current_buffers[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
            
            for i in self.periodic_devices:
                if self.offsets[i] == 0:
                    self.current_buffers[int(i), self.deadlines[i]-1] = np.random.binomial(1, self.arrival_probs[i])
        else:
            raise ValueError('traffic model not supported')

        # Set channel
        self.channel_state = np.ones(self.n_channels+1)

        self.timestep = 0
        self.discarded_packets = np.zeros(self.n_agents)
        self.received_packets = np.copy(self.current_buffers).sum(1)
        self.last_time_transmitted = np.ones(self.n_agents)
        self.last_attempts = 0
        self.successful_transmissions = 0
        self.last_feedback = 0
        self.channel_errors = 0
        self.n_collisions = 0
        self.selected_channel_qualities = 0
        self.number_selected_channel = 0

        obs = []
        for k in range(self.n_agents):
            buffer_obs = self.current_buffers[k, :self.deadlines[k]] # All values > deadline are 0 and do not provide info.
            channel_obs = np.zeros(self.n_channels + 1)
            obs.append(np.concatenate([buffer_obs, channel_obs]))
        all_buffers = np.concatenate([self.current_buffers[i, :self.deadlines[i]] for i in range(self.n_agents)])
        state = [all_buffers, self.channel_state]

        return obs, state

    def decode_signal(self, attempts_idx):
        decoded_result = np.random.binomial(1, self.channel_state[attempts_idx])
        return decoded_result
    
    def evolve_channel(self):
        change_idx = np.array([np.random.binomial(1, self.channel_switch[k]) for k in range(self.n_channels+1)])
        change_idx = change_idx.nonzero()[0]
        self.channel_state[change_idx] = 1 - self.channel_state[change_idx]

    def evolve_buffer(self, buffer):
        new_buffer = buffer[:, 1:]
        new_buffer = np.concatenate([new_buffer, np.zeros((self.n_agents,1))], axis=1)
        expired = buffer[:, 0]
        return new_buffer, expired


    def step(self, actions):
        # actions is a vector of size K with the id of the selected channel
        self.timestep += 1
        next_buffers = self.current_buffers.copy()
        self.last_time_transmitted += 1
        self.last_attempts += 1
        decoded_result = -1
        
        has_a_packet = (self.current_buffers.sum(1) > 0) * 1.
        attempts = actions * has_a_packet #In attemps, we have the channel id where the user makes a tx attempt.
        # We get the indexes of the channel attempts and the number of attempts per channel 
        channel_attempts_idx, counts = np.unique(attempts[attempts != 0], return_counts=True)
        channel_attempts_idx = channel_attempts_idx.astype(int)
        acknack = np.zeros(self.n_channels+1)
        # Setting the bad channel choices to -1. Note: good channels are set to 1.
        acknack[channel_attempts_idx] = 2 * self.channel_state[channel_attempts_idx] - 1
        self.selected_channel_qualities += (acknack > 0).sum()
        self.number_selected_channel += (acknack != 0).sum()
        
        # Setting the good channel chosen to 1 / number of attempts
        attempts_good_channel = channel_attempts_idx[self.channel_state[channel_attempts_idx] != 0]
        acknack[attempts_good_channel] = 1 / counts[self.channel_state[channel_attempts_idx] != 0]
                       
        # Good channels with only 1 user attempt
        good_channels_1_attempt = channel_attempts_idx[counts == 1]
        good_channels_1_attempt = good_channels_1_attempt[self.channel_state[good_channels_1_attempt] == 1]
        successful_users = np.where(np.isin(attempts, good_channels_1_attempt))[0]
        
        # Executing the action and increment the buffers
        for u in successful_users:
            self.successful_transmissions += 1
            self.last_time_transmitted[u] = 1.

            # Remove the decoded packet in the buffers
            col = next_buffers[u].nonzero()[0]
            next_buffers[u, col.min()] -= 1

        # We evolve the buffers and the channel
        next_buffers, expired = self.evolve_buffer(next_buffers)
        self.discarded_packets += expired
        self.evolve_channel()

        # Receive new packets
        if self.traffic_model == 'aperiodic':
            for i in range(self.n_agents):
                next_buffers[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
                self.received_packets[i] += next_buffers[i, self.deadlines[i]-1]
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.timestep % self.period == self.offsets)[0]
            for ao in active_offsets:
                next_buffers[ao, self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
                self.received_packets[ao] += next_buffers[ao, self.deadlines[ao]-1]
        
        elif self.traffic_model == 'heterogeneous':
            for i in self.aperiodic_devices:
                next_buffers[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
                self.received_packets[i] += next_buffers[i, self.deadlines[i]-1]
            
            for i in self.periodic_devices:
                if self.timestep % self.period[i] == self.offsets[i]:
                    next_buffers[int(i), self.deadlines[i]-1] = np.random.binomial(1, self.arrival_probs[i])
                    self.received_packets[i] += next_buffers[i, self.deadlines[i]-1]

        # Build observations
        obs = []
        for k in range(self.n_agents):
            buffer_obs = next_buffers[k, :self.deadlines[k]] # All values > deadline are 0 and do not provide info.
            channel_obs = acknack.copy()
            obs.append(np.concatenate([buffer_obs, channel_obs]))
        all_buffers = np.concatenate([next_buffers[i, :self.deadlines[i]] for i in range(self.n_agents)])
        state = [all_buffers, self.channel_state]
            
        rewards = np.array([len(successful_users) for _ in range(self.n_agents)])

        if self.verbose:
            print(f"Timestep {self.timestep}")
            print(f"Buffers {self.current_buffers}")
            print(f"Channels {self.channel_state}")
            print(f"Next Observation {obs}")
            print(f"Action {actions}")
            print(f"Attempts good channel {attempts_good_channel}")
            print(f"Attempts good channel 1 tx: {good_channels_1_attempt}")
            print(f'Next buffers {next_buffers}')
            print(f"Reward {rewards}")
            print(f"Received packets {self.received_packets}")
            print(f"Number of discarded packets {self.discarded_packets.sum()}")
            print("")


        if (self.timestep >= self.episode_length) :    
            done = True
        else:
            done = False

        self.current_buffers = next_buffers.copy()
        self.last_feedback = acknack

        info = {}
        return obs, state, rewards, done, info


    def compute_jains(self):
        urllc_scores = []
        for k in range(self.n_agents):
            if self.received_packets[k] > 0:
                urllc_scores.append(1 - self.discarded_packets[k] / self.received_packets[k])
            else:
                urllc_scores.append(1)
        urllc_scores = np.array(urllc_scores)
        jains = urllc_scores.sum() ** 2 / self.n_agents / (urllc_scores ** 2).sum()
        return jains

    def compute_urllc(self):
        urllc_scores = 1 - self.discarded_packets.sum() / self.received_packets.sum()
        return urllc_scores
    
    def compute_channel_score(self):
        if self.number_selected_channel != 0:
            return self.selected_channel_qualities / self.number_selected_channel
        else:
            return 1