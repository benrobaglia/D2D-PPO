import numpy as np
from gym import spaces

class CombinatorialEnv:

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
                collision_type="pessimistic",
                homogeneous_size=False,
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
        self.collision_type = collision_type
        # iDQN is implemented to handle observations of agents with the same size
        # Thus if homogeneous_size == True, observations are padded.
        self.homogeneous_size = homogeneous_size
        self.reward_type = reward_type
        self.periodic_devices = periodic_devices
        self.aperiodic_devices = [i for i in range(self.n_agents) if i not in periodic_devices]

        # Channel
        if channel_switch is None:
            self.channel_switch = np.zeros((self.n_agents, self.n_channels))
        else:
            self.channel_switch = channel_switch

        if not self.homogeneous_size:
            # Observation of device k: buffer of size the deadline + channel state + last feedback
            self.observation_space = spaces.Tuple([spaces.Box(low=-float('inf'), high=float('inf'),
                                                            shape=(self.deadlines[k] + 2 * self.n_channels,)) for k in range(self.n_agents)])
        else:
            self.observation_space = spaces.Tuple([spaces.Box(low=-float('inf'), high=float('inf'),
                                                            shape=(self.deadlines.max() + 2 * self.n_channels,)) for k in range(self.n_agents)])
        # If the first dimension is chosen, no packet is transmitted and the agent remains idle.
        self.action_space = spaces.Tuple([spaces.MultiBinary(self.n_channels) for _ in range(self.n_agents)])

        self.state_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                                        shape=(self.deadlines.sum() + self.n_channels*(self.n_agents + 1),))


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
        self.channel_state = np.ones((self.n_agents, self.n_channels))

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
            if not self.homogeneous_size:
                buffer_obs = self.current_buffers[k, :self.deadlines[k]] # All values > deadline are 0 and do not provide info.
            else:
                buffer_obs = self.current_buffers[k]
            channel_obs = np.ones(self.n_channels)
            obs.append(np.concatenate([buffer_obs, channel_obs, channel_obs]))
        all_buffers = np.concatenate([self.current_buffers[i, :self.deadlines[i]] for i in range(self.n_agents)])
        all_channels = self.channel_state.reshape(-1)
        state = [all_buffers, all_channels, channel_obs]

        return obs, state
    
    def evolve_channel(self):
        change = np.random.binomial(1, self.channel_switch)
        self.channel_state = np.abs(self.channel_state - change)

    def evolve_buffer(self, buffer):
        new_buffer = buffer[:, 1:]
        new_buffer = np.concatenate([new_buffer, np.zeros((self.n_agents,1))], axis=1)
        expired = buffer[:, 0]
        return new_buffer, expired


    def step(self, actions):
        # actions is a binary matrix of size K x n_channels
        self.timestep += 1
        next_buffers = self.current_buffers.copy()
        self.last_time_transmitted += 1
        self.last_attempts += 1
        decoded_result = -1
        
        has_a_packet = (self.current_buffers.sum(1) > 0) * 1.
        has_a_packet_mat = np.stack([has_a_packet for _ in range(actions.shape[1])]).T
        attempts = actions * has_a_packet_mat
        attempts_good_channels = attempts * self.channel_state
        
        # Get all the channels that have been selected by all users
        selected_channels = (attempts.sum(0) > 0) * 1
        # selected_channels_mat = np.tile(selected_channels, (self.n_agents, 1))

        # Get channel obs
        channel_obs = self.channel_state.copy()

        # Get the number of users that selected the channels
        n_users_per_channel = attempts.sum(0)

        # print(acknack)
        # print(mask)
        # print(self.channel_state)
        # print(selected_channels)
        # print(n_users_per_channel)
        acknack = np.zeros(self.n_channels) - 1
        acknack[(attempts_good_channels.sum(0) == 1) & (n_users_per_channel == 1)] = 1
        acknack[n_users_per_channel == 0] = 0
        acknack_mat = np.tile(acknack, (self.n_agents, 1))
        
        successful_attempts = ((acknack_mat * attempts_good_channels) == 1)
        successful_users = np.unique(successful_attempts.nonzero()[0])

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
            if not self.homogeneous_size:
                buffer_obs = next_buffers[k, :self.deadlines[k]] # All values > deadline are 0 and do not provide info.
            else:
                buffer_obs = next_buffers[k]
            channel_obs_k = channel_obs[k]
            obs.append(np.concatenate([buffer_obs, channel_obs_k, acknack]))
        all_buffers = np.concatenate([next_buffers[i, :self.deadlines[i]] for i in range(self.n_agents)])
        all_channels = self.channel_state.reshape(-1)
        state = [all_buffers, all_channels, acknack]
            
        rewards = np.array([len(successful_users) for _ in range(self.n_agents)])

        if self.verbose:
            print(f"Timestep {self.timestep}")
            print(f"Buffers {self.current_buffers}")
            print(f"Channel state {channel_obs}")
            print(f"Attempts x good channel: {attempts_good_channels}")
            print(f"Next Observation {obs}")
            print(f"Action {actions}")
            print(f"Attempts {attempts}")
            print(f"ACK/NACK {acknack}")
            print(f"Selected channels {selected_channels}")
            print(f"N users per channel {n_users_per_channel}")
            print(f"Successful users: {successful_users}")
            print(f'Next buffers {next_buffers}')
            print(f"Next Channels {self.channel_state}")
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