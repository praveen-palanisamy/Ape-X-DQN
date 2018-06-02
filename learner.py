#!/usr/bin/env python
import torch
from duelling_network import DuellingDQN

class Learner(object):
    def __init__(self, env_conf, learner_params, shared_state):
        state_shape = env_conf['state_shape']
        action_dim = env_conf['action_dim']
        self.shared_state = shared_state
        self.Q = DuellingDQN(state_shape, action_dim)
        self.Q_double = DuellingDQN(state_shape, action_dim)
        if learner_params['load_saved_state']:
            try:
                saved_state = torch.load(learner_params['load_saved_state'])
                self.Q.load_state_dict(saved_state['Q_state'])
            except FileNotFoundError:
                print("WARNING: No trained model found. Training from scratch")
        self.shared_state["Q_state_dict"] = self.Q.state_dict()

    def learn(self, T):
        for t in range(T):
            id, prioritized_xp_batch = replay_memory.sample(self.params['replay_sample_size'])
            loss = self.compute_loss(prioritized_xp_batch)
            self.update_Q(loss)
            self.shared_state['Q_state_dict'] = self.Q.state_dict()
            priorities = self.compute_priorities(prioritized_xp_batch)
            replay_memory.set_priority(id, priorities)

            if t % self.params['remove_old_xp_freq'] == 0:
                replay_memory.cleanup_old_xp()
