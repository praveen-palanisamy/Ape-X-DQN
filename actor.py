#!/usr/bin/env python
import torch
from duelling_network import DuellingDQN
from env import make_local_env

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
        self.local_experience_buffer = list()

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





