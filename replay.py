from collections import namedtuple
import numpy as np


N_Step_Transition = namedtuple('N_Step_Transition', ['S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'])


class ReplayMemory(object):
    def __init__(self, soft_capacity, params):
        self.soft_capacity = soft_capacity
        self.memory = list()
        self.mem_idx = 0
        self.counter = 0
        self.alpha = params['priority_exponent']
        self.priorities = dict()
        self.sample_probabilities = dict()

    def update_sample_probabilities(self):
        """
        Updates the probability of sampling an experience from the replay memory
        :return:
        """
        priorities = self.priorities
        prob = [p**self.alpha/ sum(priorities.values())  for p in priorities.values()]
        prob /= sum(prob)
        self.sample_probabilities.update({k:v for k in list(priorities) for v in prob})
        # Let the probabilities sum to 1
        sum_of_prob = sum(self.sample_probabilities.values())
        for k in self.sample_probabilities.keys():
            self.sample_probabilities[k] /= sum_of_prob

    def set_priorities(self, new_priorities):
        """
        Updates the priorities of experience using the key that uniquely identifies an experience.
        If a key does not exist, it is added. If a key already exists, the priority value is updated/overwritten.
        Whenever the priorities are altered/added, the sample probabilities are also updated.
        :param priorities: A dictionary with experience_key: priority_value key-value pairs
        :return: None
        """
        self.priorities.update(new_priorities)
        # Update the sample probabilities as well
        self.update_sample_probabilities()

    def sample(self, sample_size):
        """
        Returns a batch of experiences sampled from the replay memory based on the sampling probability calculated using
        the experience priority
        :param sample_size: Size of the batch to be sampled from the prioritized replay buffer
        :return: A list of N_Step_Transition objects
        """
        mem = N_Step_Transition(*zip(*self.memory))
        sampled_keys = [np.random.choice(list(self.priorities.keys()), p=list(self.sample_probabilities.values()))
                        for _ in range(sample_size) ]
        batch_xp = [N_Step_Transition(S, A, R, G, qt, Sn, qn, key) for k in sampled_keys
                    for S, A, R, G, qt, Sn, qn, key in zip(mem.S_t, mem.A_t, mem.R_ttpB, mem.Gamma_ttpB,
                                                           mem.qS_t, mem.S_tpn, mem.qS_tpn, mem.key) if key == k]
        return batch_xp

    def add(self, priorities, xp_batch):
        """
        Adds batches of experiences and priorities to the replay memory
        :param priorities: Priorities of the experiences in xp_batch
        :param xp_batch: List of experiences of type N_Step_Transitions
        :return:
        """
        # Add the new experience data to replay memory
        [self.memory.append(xp) for xp in xp_batch]
        # Set the initial priorities of the new experiences using set_priorities which also takes care of updating prob
        self.set_priorities(priorities)

    def remove_to_fit(self):
        """
        Method to remove replay memory data above the soft capacity threshold. The experiences are removed in FIFO order
        This method is called by the learner periodically
        :return:
        """
        if self.size() > self.soft_capacity:
            num_excess_data = self.size() - self.soft_capacity
            # FIFO
            del self.memory[: num_excess_data]

    def size(self):
        return len(self.memory)

