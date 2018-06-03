from collections import namedtuple
import numpy as np


N_Step_Transition = namedtuple('N_Step_Transition', ['S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'])
Prioritized_N_Step_Transition = namedtuple('Prioritized_N_Step_Transition', ['St', 'At', 'R_ttpB', 'Gamma_ttpB',
                                                                             'S_tpn', 'priority', 'key'])
class ReplayMemory(object):
    def __init__(self, soft_capacity, params):
        self.soft_capacity = soft_capacity
        self.memory = list()
        self.mem_idx = 0
        self.counter = 0
        self.alpha = params['priority_exponent']
        self.priorities = dict()

    def update_sample_probability(self):
        priorities = self.priorities
        prob = [p**self.alpha/ sum(priorities.values())  for p in priorities.values()]
        self.prob = {k:v for k in priorities.keys() for v in prob}

    def sample(self, sample_size):
        """
        Returns a batch of experiences sampled from the replay memory based on the sampling probability calculated using
        the experience priority
        :param sample_size: Size of the batch to be sampled from the prioritized replay buffer
        :return: A list of N_Step_Transition objects
        """
        sampled_keys = [np.random.choice(list(self.priorities.keys(), list(self.probs.values())))
                        for _ in range(sample_size) ]
        batch_xp = [N_Step_Transition(S, A, R, G, qt, Sn, qn, key) for k in sampled_keys
                    for S, A, R, G, qt, Sn, qn, key in zip(self.memory.S_t, self.memory.A_t, self.memory.R_ttpB,
                                                self.memory.Gamma_ttpB, self.memory.qS_t, self.memory.S_tpn,
                                                self.memory.qS_tpn, self.memory.key) if key == k]
        return batch_xp


    def set_priorities(self, new_priorities):
        """
        Updates the priorities of experience using the key that uniquely identifies an experience.
        If a key does not exist, it is added. If a key already exists, the priority value is updated/overwritten
        :param priorities: A dictionary with experience_key: priority_value key-value pairs
        :return: None
        """
        self.priorities.update(new_priorities)

    def remove_to_fit(self):
        if self.size > self.soft_capacity:
            num_excess_data = self.size - self.soft_capacity
            del self.memory[: num_excess_data]

    @property
    def size(self):
        return len(self.memory)

