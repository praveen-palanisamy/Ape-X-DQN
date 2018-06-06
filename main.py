#!/usr/bin/env python
import torch
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
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

BaseManager.register("Memory", ReplayMemory)


def add_experience_to_replay_mem(shared_mem, replay_mem):
    while 1:
        while shared_mem.qsize() or not shared_mem.empty():
            priorities, xp_batch = shared_mem.get()
            replay_mem.add(priorities, xp_batch)


if __name__ =="__main__":
    params = json.load(open(args.params_file, 'r'))
    env_conf = params['env_conf']
    actor_params = params["Actor"]
    learner_params = params["Learner"]
    replay_params = params["Replay_Memory"]

    dummy_q = DuellingDQN(tuple(env_conf['state_shape']), env_conf['action_dim'])
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_mem = mp_manager.Queue()
    ReplayManager = BaseManager()
    ReplayManager.start()
    replay_mem = ReplayManager.Memory(1000,  replay_params)
    print("Main: RPM.size:", replay_mem.size())

    #  TODO: Start Actors in separate proc
    actor = Actor(1, env_conf, shared_state, shared_mem, actor_params)
    actor_procs = mp.Process(target=actor.gather_experience, args=(1110,))
    actor_procs.start()

    # TODO: Run a routine in a separate proc to fetch/pre-fetch shared_replay_mem onto the ReplayBuffer for learner's use
    replay_mem_proc = mp.Process(target=add_experience_to_replay_mem, args=(shared_mem, replay_mem))
    replay_mem_proc.start()
    actor_procs.join()
    replay_mem_proc.join()


    print("Main: replay_mem.size:", shared_mem.qsize())