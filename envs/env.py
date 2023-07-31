import numpy as np
from gym import spaces

class D2DEnv:
    def __init__(self,
                n_agents,
                deadlines,
                lbdas,
                period=5,
                arrival_probs=None,
                offsets=None,
                episode_length=100,
                traffic_model='aperiodic',
                periodic_devices=[],
                reward_type=0,
                channel_switch=0.2,
                channel_decoding=0.8,
                neighbourhoods=None, # Neighbourhoods is a list of size n_agents with the indices of the neighbours for each agent.
                verbose=False): 
        
        self.verbose = verbose
        self.n_agents = n_agents
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
        self.channel_switch = channel_switch
        self.channel_decoding = channel_decoding

        if neighbourhoods is None:
            self.neighbourhoods = [[k] for k in range(self.n_agents)]
        else:
            self.neighbourhoods = neighbourhoods
        # Observation of device k: buffer of size the deadline + channel state + last feedback for all k's neighbours.
        self.observation_space = spaces.Tuple([spaces.Box(low=-float('inf'), high=float('inf'),
                                                           shape=(self.deadlines[self.neighbourhoods[k]].sum() + len(self.neighbourhoods[k]) + 1,)) for k in range(self.n_agents)])
        self.action_space = spaces.Tuple([spaces.Discrete(2) for _ in range(self.n_agents)])

    def reset(self):
        self.current_buffers = np.zeros((self.n_agents, np.max(self.deadlines)))      # The buffer status of all devices.
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
        # self.channel_state = np.random.choice([self.channel_decoding, 1-self.channel_decoding], self.n_agents)
        self.channel_state = np.ones(self.n_agents)

        self.timestep = 0
        self.discarded_packets = np.zeros(self.n_agents)
        self.received_packets = np.copy(self.current_buffers).sum(1)
        self.last_time_transmitted = np.ones(self.n_agents)
        self.last_attempts = 0
        self.successful_transmissions = 0
        self.last_feedback = 0
        self.channel_errors = 0
        self.n_collisions = 0

        # Compute observations: an observation of an agent k is the concatenation of the buffers and channels of its neighbours
        obs = []
        for k in range(self.n_agents):
            buffer_obs = np.concatenate([self.current_buffers[i, :self.deadlines[i]] for i in self.neighbourhoods[k]]) # All values > deadline are 0 and do not provide info.
            channel_obs = self.channel_state[self.neighbourhoods[k]]
            obs.append(np.concatenate([buffer_obs, channel_obs, np.copy([self.last_feedback])]))

        state = [self.current_buffers.copy(), self.channel_state]

        return obs, state

    def decode_signal(self, attempts_idx):
        decoded_result = np.random.binomial(1, self.channel_state[attempts_idx])
        return decoded_result
    
    def evolve_channel(self):
        change_idx = np.random.binomial(1, self.channel_switch, self.n_agents).nonzero()[0]
        self.channel_state[change_idx] = 1 - self.channel_state[change_idx]

    def evolve_buffer(self, buffer):
        new_buffer = buffer[:, 1:]
        new_buffer = np.concatenate([new_buffer, np.zeros((self.n_agents,1))], axis=1)
        expired = buffer[:, 0]
        return new_buffer, expired


    def step(self, actions):
        self.timestep += 1
        next_buffers = self.current_buffers.copy()

        self.last_time_transmitted += 1
        self.last_attempts += 1
        decoded_result = -1
        has_a_packet = (self.current_buffers.sum(1) > 0) * 1.
        attempts = actions * has_a_packet
        n_attempts = attempts.sum()

        # Executing the action and increment the buffers
        if n_attempts == 1:
            try:
                attempts_idx = attempts.nonzero()[0].item()
            except:
                raise ValueError(f"Only 1 device should try to transmit. attempts: {attempts}")

            decoded_result = self.decode_signal(attempts_idx)
            if decoded_result:
                ack_nack = 1
                self.successful_transmissions += 1
                self.last_time_transmitted[attempts_idx] = 1.

                # Remove the decoded packet in the buffers
                col = next_buffers[attempts_idx].nonzero()[0]
                next_buffers[attempts_idx, col.min()] -= 1
            else:
                ack_nack = 0
                self.channel_errors += 1
        elif n_attempts > 1:
            ack_nack = -1
            self.n_collisions += 1
        elif n_attempts == 0:
            ack_nack = 0
        else:
            raise ValueError(f'rewards sum error. Not possible. attempts: {attempts}')     

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
            buffer_obs = np.concatenate([next_buffers[i, :self.deadlines[i]] for i in self.neighbourhoods[k]]) # All values > deadline are 0 and do not provide info.
            channel_obs = self.channel_state[self.neighbourhoods[k]]
            obs.append(np.concatenate([buffer_obs, channel_obs, np.copy([ack_nack])]))

        state = [next_buffers.copy(), self.channel_state.copy(), ack_nack]
        rewards = np.zeros(self.n_agents) + ack_nack

        if self.verbose:
            print(f"Timestep {self.timestep}")
            print(f"Buffers {self.current_buffers}")
            print(f"Channels {self.channel_state}")
            print(f"Next Observation {obs}")
            print(f"Action {actions}")
            print(f"Decoded {decoded_result}")
            print(f"Channel errors: {self.channel_errors}")
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
        self.last_feedback = ack_nack

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
