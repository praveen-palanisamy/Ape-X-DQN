#!/usr/bin/env python
import torch
import time
import numpy as np
from collections import namedtuple
from duelling_network import DuellingDQN

N_Step_Transition = namedtuple('N_Step_Transition', ['S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'])

class Learner(object):
    def __init__(self, env_conf, learner_params, shared_state, shared_replay_memory):
        self.state_shape = env_conf['state_shape']
        action_dim = env_conf['action_dim']
        self.params = learner_params
        self.shared_state = shared_state
        self.Q = DuellingDQN(self.state_shape, action_dim)
        self.Q_double = DuellingDQN(self.state_shape, action_dim)  # Target Q network which is slow moving replica of self.Q
        if self.params['load_saved_state']:
            try:
                saved_state = torch.load(self.params['load_saved_state'])
                self.Q.load_state_dict(saved_state['Q_state'])
            except FileNotFoundError:
                print("WARNING: No trained model found. Training from scratch")
        self.shared_state["Q_state_dict"] = self.Q.state_dict()
        self.replay_memory = shared_replay_memory
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=0.00025 / 4, weight_decay=0.95, eps=1.5e-7)
        self.num_q_updates = 0

    def compute_loss_and_priorities(self, xp_batch):
        """
        Computes the double-Q learning loss and the proportional experience priorities.
        :param xp_batch: list of experiences of type N_Step_Transition
        :return: double-Q learning loss and the proportional experience priorities
        """
        n_step_transitions = N_Step_Transition(*zip(*xp_batch))
        # Convert tuple to numpy array; Convert observations(S_t and S_tpn) to c x w x h torch Tensors (aka Variable)
        S_t = torch.from_numpy(np.array(n_step_transitions.S_t)).float().requires_grad_(True)
        S_tpn = torch.from_numpy(np.array(n_step_transitions.S_tpn)).float().requires_grad_(True)
        rew_t_to_tpB = np.array(n_step_transitions.R_ttpB)
        gamma_t_to_tpB = np.array(n_step_transitions.Gamma_ttpB)
        A_t = np.array(n_step_transitions.A_t)

        with torch.no_grad():
            G_t = rew_t_to_tpB + gamma_t_to_tpB * \
                             self.Q_double(S_tpn)[2].gather(1, torch.argmax(self.Q(S_tpn)[2], 1).view(-1, 1)).squeeze()
        Q_S_A = self.Q(S_t)[2].gather(1, torch.from_numpy(A_t).reshape(-1, 1)).squeeze()
        batch_td_error = G_t.float() - Q_S_A
        loss = 1/2 * (batch_td_error)**2
        # Compute the new priorities of the experience
        priorities = {k: v for k in xp_batch.keys for v in abs(batch_td_error)}

        return loss, priorities

    def update_Q(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_q_updates += 1

        if self.num_q_updates % self.params['q_target_sync_freq']:
            self.Q_double.load_state_dict(self.Q.state_dict())

    def learn(self, T):
        while self.replay_memory.size() <=  self.params["min_replay_mem_size"]:
            print("rpm.size:", self.replay_memory.size(), "Waiting to get at least", self.params['min_replay_mem_size'])
            time.sleep(1)
        for t in range(T):
            print("t=", t, "have enough items in replay mem. Starting to learn")
            # 4. Sample a prioritized batch of transitions
            prioritized_xp_batch = self.replay_memory.sample(int(self.params['replay_sample_size']))
            print("p_xp_batch size:", len(prioritized_xp_batch) )
            # 5. & 7. Apply double-Q learning rule, compute loss and experience priorities
            loss, priorities = self.compute_loss_and_priorities(prioritized_xp_batch)
            # 6. Update parameters of the Q network(s)
            self.update_Q(loss)
            self.shared_state['Q_state_dict'] = self.Q.state_dict()
            # 8. Update priorities
            self.replay_memory.set_priority(id, priorities)

            # 9. Periodically remove old experience from replay memory
            if t % self.params['remove_old_xp_freq'] == 0:
                self.replay_memory.cleanup_old_xp()
