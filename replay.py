from collections import namedtuple
import numpy as np

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
        sampled_keys = np.random.choice(list(self.priorities.keys(), list(self.probs.values())))


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

