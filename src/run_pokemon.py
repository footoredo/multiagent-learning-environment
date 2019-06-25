from env.matrix_env import MatrixEnv
from env.pokemon_env import PokemonEnv, get_pokemon, get_move_number, PokemonInstance
from controller.naive_controller import NaiveController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent import PPOAgent
from agent.mlp_policy import MLPPolicy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logger
import numpy as np


def make_dummy_agent(observation_space, action_space, handlers):
    return DummyAgent(2)


def make_infant_agent(observation_space, action_space, handlers):
    return InfantAgent(2, handlers)


def make_self_play_agent(observation_space, action_space, handlers):
    return SelfPlayAgent(2, handlers)


ppo_agent_cnt = 0


def get_make_ppo_agent(timesteps_per_actorbatch, max_episodes):
    def make_ppo_agent(observation_space, action_space, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=16, num_hid_layers=2)

        global ppo_agent_cnt
        agent = PPOAgent(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                         ob_space=observation_space, ac_space=action_space, handlers=handlers,
                         timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0.0,
                         optim_epochs=1, optim_stepsize=5e-5,
                         gamma=1., lam=0.95, max_episodes=max_episodes, schedule="wolf_adv")
        ppo_agent_cnt += 1
        return agent
    return make_ppo_agent


def debugger(infos):
    # print(infos[])
    print(sum(info[0] for info in infos) / len(infos),
          sum(info[1] for info in infos) / len(infos))


if __name__ == "__main__":
    train = True

    # _kyogre = get_pokemon("Primal Kyogre")
    # kyogre = PokemonInstance(_kyogre, moves=[
    #     get_move_number("Water Spout"),
    #     get_move_number("Origin Pulse")
    # ], ivs=[31, 0, 31, 31, 31, 31], evs=[4, 0, 0, 252, 0, 252], nature=(3, 1), level=50)
    #
    # _groudon = get_pokemon("Primal Groudon")
    # groudon = PokemonInstance(_groudon, moves=[
    #     get_move_number("Precipice Blades"),
    #     get_move_number("Fire Punch")
    # ], ivs=[31, 31, 31, 31, 31, 31], evs=[252, 252, 0, 0, 4, 0], nature=(1, 3), level=50)
    #
    # _caterpie = get_pokemon("Caterpie")
    # caterpie = PokemonInstance(_caterpie, moves=[
    #     get_move_number("Protect"),
    # ], ivs=[0, 0, 0, 0, 0, 0], evs=[0, 0, 0, 0, 0, 0], nature=(1, 1), level=1)

    # _rowlet = get_pokemon("Rowlet")
    # rowlet = PokemonInstance(_rowlet, moves=[
    #     get_move_number("Tackle"),
    #     get_move_number("Growl")
    # ], ivs=[31, 31, 31, 31, 31, 0], evs=[0, 0, 0, 0, 0, 0], nature=(3, 5), level=5)
    #
    # _popplio = get_pokemon("Popplio")
    # popplio = PokemonInstance(_popplio, moves=[
    #     get_move_number("Pound")
    # ], ivs=[31, 31, 31, 0, 31, 31], evs=[0, 252, 0, 0, 0, 0], nature=(1, 3), level=5)

    # _groudon = get_pokemon("Primal Groudon")
    # groudonA = PokemonInstance(_groudon, moves=[
    #     get_move_number("Sword Dance"),
    #     get_move_number("Fire Punch")
    # ], ivs=[31, 31, 31, 31, 31, 31], evs=[252, 252, 0, 0, 4, 0], nature=(1, 3), level=50, nickname="GroudonA")
    # groudonB = PokemonInstance(_groudon, moves=[
    #     get_move_number("Fire Punch")
    # ], ivs=[31, 31, 31, 31, 31, 31], evs=[252, 252, 0, 0, 0, 4], nature=(1, 3), level=50, nickname="GroudonB")

    env = PokemonEnv(groudonA, groudonB)

    controller = NaiveController(env, [get_make_ppo_agent(16, 64), get_make_ppo_agent(16, 64)])
    controller.train(max_steps=50000, policy_store_every=None, test_every=None, show_every=10, test_max_steps=100)
