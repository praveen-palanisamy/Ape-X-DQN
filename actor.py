#!/usr/bin/env python
import torch
from collections import namedtuple
from duelling_network import DuellingDQN
from env import make_local_env

Transition = namedtuple('Transition', ['s', 'a', 'r', 'gamma', 'q'])
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
        self.idx = 0
        self.capacity = n

    def update_buffer(self):
        """
        Updates the accumulated per-step discount and the partial return for every item in the buffer. This should be
        called after every new transition is added to the buffer
        :return: None
        """
        for i in range(self.B):
            for k in range(1, self.B):
                self.buffer[i].gamma = self.buffer[i].gamma ** k
                self.buffer[i].r += self.gamma * self.buffer[i+1].r

    def calculate_exp_priorities(self, n_step_transitions):
        #  Calculate the absolute n-step TD errors
        n_step_td_target = n_step_transitions.rewards + n_step_transitions.gammas * n_step_transitions.qS_tpn
        n_step_td_error = n_step_td_target - n_step_transitions.qS_t
        priorities = abs(n_step_td_error)


    def add(self, data):
        """
        Add transition data to the Experience Buffer and update the accumulated per-step discounts and partial returns
        :param data: tuple containing a transition data of type Transition(s, a, r, gamma, q)
        :return: None
        """
        if self.idx  + 1 < self.capacity:
            self.idx += 1
            self.buffer[self.idx] = data
            self.update_buffer()  #  calculate the accumulated per-step disc & partial return for all entries
        else:  # Buffer has reached its capacity, n
            #  Construct the n-step transition
            n_step_transition = N_Step_Transition(*self.buffer[0], data['s'], data['q'])
            #  Put the n_step_transition into a Queue
            #  Calculate the priorities in batch
            #  Send the  n_step_transitionS to the global replay memory


    @property
    def B(self):
        """
        The current size of buffer. B follows the same notation as in the Ape-X paper(TODO: insert link to paper)
        :return: The current size of the buffer
        """
        return len(self.buffer)


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





