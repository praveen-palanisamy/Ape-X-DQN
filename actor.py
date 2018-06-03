#!/usr/bin/env python
import torch
from collections import namedtuple
from duelling_network import DuellingDQN
from env import make_local_env

Transition = namedtuple('Transition', ['S', 'A', 'R', 'Gamma', 'q'])
N_Step_Transition = namedtuple('N_Step_Transition', ['St', 'At', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn'])
Prioritized_N_Step_Transition = namedtuple('Prioritized_N_Step_Transition', ['St', 'At', 'R_ttpB', 'Gamma_ttpB',
                                                                             'S_tpn', 'priority', 'key'])
class ExperienceBuffer(object):
    def __init__(self, n):
        """
        Implements a circular/ring buffer to store n-step transition data used by the actor
        :param n:
        """
        self.buffer = list()
        self.idx = -1
        self.capacity = n
        self.local_memory = list()  #  To store n-step transitions b4 they r batched, prioritized and sent to replay mem
        self.gamma = 0.99

    def update_buffer(self):
        """
        Updates the accumulated per-step discount and the partial return for every item in the buffer. This should be
        called after every new transition is added to the buffer
        :return: None
        """
        for i in range(self.B - 1):
            R = self.buffer[i].R
            Gamma = self.buffer[i].Gamma
            for k in range(1, self.B):
                Gamma = self.buffer[i].Gamma ** k
                R += self.gamma * self.buffer[i+1].R
            self.buffer[i] = Transition(self.buffer[i].S, self.buffer[i].A, R, Gamma, self.buffer[i].q)

    def add(self, data):
        """
        Add transition data to the Experience Buffer and calls update_buffer
        :param data: tuple containing a transition data of type Transition(s, a, r, gamma, q)
        :return: None
        """
        if self.idx  + 1 <= self.capacity:
            self.idx += 1
            self.buffer.append(None)
            self.buffer[self.idx] = data
            self.update_buffer()  #  calculate the accumulated per-step disc & partial return for all entries
        else:  # Buffer has reached its capacity, n
            #  Construct the n-step transition
            print("self.buffer.len:", len(self.buffer))
            n_step_transition = N_Step_Transition(*self.buffer[0], data.S, data.q)
            #  Put the n_step_transition into a local memory store
            self.local_memory.append(n_step_transition)
            #  Free-up the buffer
            self.buffer.clear()
            self.idx = -1

    def get(self, batch_size):
        assert batch_size <= self.size, "Requested n-step transitions batch size is more than available"
        batch_of_n_step_transitions = self.local_memory[: batch_size]
        del self.local_memory[: batch_size]
        return batch_of_n_step_transitions

    @property
    def B(self):
        """
        The current size of buffer. B follows the same notation as in the Ape-X paper(TODO: insert link to paper)
        :return: The current size of the buffer
        """
        return len(self.buffer)

    @property
    def size(self):
        """
        The current size of the local experience memory
        :return:
        """
        return len(self.local_memory)


class Actor(object):
    def __init__(self, env_conf, shared_state, actor_params):
        state_shape = env_conf['state_shape']
        action_dim = env_conf['action_dim']
        self.params = actor_params
        self.shared_state = shared_state
        self.Q = DuellingDQN(state_shape, action_dim)
        self.Q.load_state_dict(shared_state["Q_state_dict"])
        self.env = make_local_env(env_conf['name'])
        self.policy = self.psilon_greedy_policy
        self.local_experience_buffer = ExperienceBuffer(self.params["local_experience_buffer_capacity"])

    def epsilon_greedy_policy(self, obs):
        pass

    def gather_experience(self, T):
        obs = self.env.reset()
        for t in range(T):
            action = self.policy(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.local_experience_buffer.add(obs, action, reward, next_obs)

            if self.local_experience_buffer.size() >= self.params['experience_count_threshold']:
                experience_batch = self.local_experience_buffer.get(self.params['experience_count_threshold'])
                priority = self.compute_priorities(experience_batch)
                self.shared_state["global_replay_memory"].add(experience_batch, priority)

            if t % self.params['Q_network_sync_freq'] == 0:
                self.Q.load_state_dict(self.shared_state["Q_state_dict"])





