#!/usr/bin/env python
import torch
import torch.multiprocessing as mp
import json
from replay import ReplayMemory
from actor import Actor
from duelling_network import DuellingDQN
from argparse import ArgumentParser

arg_parser = ArgumentParser(prog="main.py")
arg_parser.add_argument("--params-file", default="parameters.json", type=str,
                    help="Path to json file defining the parameters for the Actor, Learner and Replay memory",
                    metavar="PARAMSFILE")
args = arg_parser.parse_args()


if __name__ =="__main__":
    params = json.load(open(args.params_file, 'r'))
    env_conf = params['env_conf']
    actor_params = params["Actor"]
    learner_params = params["Learner"]
    replay_params = params["Replay_Memory"]

    dummy_q = DuellingDQN(tuple(env_conf['state_shape']), env_conf['action_dim'])
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_state["Q_state_dict"] = dummy_q.state_dict()
    shared_replay_mem = mp_manager.Queue()
    actor = Actor(1, env_conf, shared_state, shared_replay_mem, actor_params)
    actor.gather_experience(11110)
    print("Main: replay_mem.size:", shared_replay_mem.qsize())