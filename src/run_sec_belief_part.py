from env.matrix_env import MatrixEnv
from env.belief_security_env_vn import BeliefSecurityEnv
from controller.belief_controller import BeliefController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent_backup import PPOAgent
from agent.ppo_agent_stacked_v3 import PPOAgentStacked
from agent.pac_agent import PACAgent
from agent.mlp_policy import MLPPolicy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from decimal import Decimal
from common.path_utils import *
import joblib
import argparse


def make_dummy_agent(observation_space, action_space, handlers):
    return DummyAgent(2)


def make_infant_agent(observation_space, action_space, handlers):
    return InfantAgent(2, handlers)


def make_self_play_agent(observation_space, action_space, handlers):
    return SelfPlayAgent(2, handlers)


ppo_agent_cnt = 0
pac_agent_cnt = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run security game.")

    parser.add_argument('--seed', type=str)
    parser.add_argument('--other', type=str, default="default")
    parser.add_argument('--agent', type=str, default="ppo")
    parser.add_argument('--n-slots', type=int, default=2)
    parser.add_argument('--n-types', type=int, default=2)
    parser.add_argument('--n-rounds', type=int, default=2)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--prior', type=float, nargs='+')
    parser.add_argument('--random-prior', action="store_true")
    parser.add_argument('--reset', action="store_true")
    parser.add_argument('--zero-sum', action="store_true")
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--schedule', type=str, nargs="+", default="constant")
    parser.add_argument('--opponent', type=str, nargs="+", default="latest")
    parser.add_argument('--test-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--reset-every', type=int)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--load-step', type=int)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--network-width', type=int, default=32)
    parser.add_argument('--network-depth', type=int, default=1)
    parser.add_argument('--test-steps', type=int, default=1000)
    parser.add_argument('--timesteps-per-batch', type=int, default=100)
    parser.add_argument('--iterations-per-round', type=int, default=1)
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--sub-load-path', type=str)
    parser.add_argument('--vn-load-path', type=str)
    parser.add_argument('--save-dir', type=str)

    return parser.parse_args()


def get_make_ppo_agent(reset_every, timesteps_per_actorbatch, max_iterations, beta1, n_rounds):
    def make_ppo_agent(observation_space, action_space, steps_per_round, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            # init_ob_space, info_ob_space = ob_space
            # return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
            #                  hid_size=256, num_hid_layers=4)
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=network_width, num_hid_layers=network_depth)

        global ppo_agent_cnt
        ob_space = observation_space
        agent = PPOAgentStacked(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                                ob_space=ob_space, ac_space=action_space,
                                handlers=handlers, reset_every=reset_every,
                                timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0,
                                beta1=beta1, n_rounds=n_rounds,
                                optim_epochs=1, optim_stepsize=learning_rate,
                                gamma=0.99, lam=0.95, max_iters=max_iterations,
                                schedule=schedule, opponent=opponent)
        ppo_agent_cnt += 1
        return agent
    return make_ppo_agent


def get_make_pac_agent(timesteps_per_actorbatch, max_iterations):
    def make_pac_agent(observation_space, action_space, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            # return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
            #                  hid_size=256, num_hid_layers=4)
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=network_width, num_hid_layers=network_depth)

        global pac_agent_cnt
        agnt = PACAgent(name="pac_agent_%d" % pac_agent_cnt, policy_fn=policy,
                        ob_space=observation_space, ac_space=action_space, handlers=handlers,
                        timesteps_per_actorbatch=timesteps_per_actorbatch,
                        optim_epochs=1, optim_stepsize=learning_rate,
                        gamma=0.99, max_iters=max_iterations,
                        schedule=schedule, opponent=opponent)
        pac_agent_cnt += 1
        return agnt
    return make_pac_agent


def debugger(infos):
    # print(infos[])
    print(sum(info[0] for info in infos) / len(infos),
          sum(info[1] for info in infos) / len(infos))


if __name__ == "__main__":

    def argument_to_tuple(argument):
        if type(argument) == list and len(argument) == 1:
            return argument[0]
        elif type(argument) == str:
            return argument
        else:
            parameters = list(map(float, argument[1:]))
            return tuple(argument[:1] + parameters)

    args = parse_args()

    seed = args.seed
    # seed = "benchmark"
    agent = args.agent
    n_slots = args.n_slots
    n_types = args.n_types
    n_rounds = args.n_rounds
    prior = args.prior or [1 / n_types for _ in range(n_types)]
    reset = args.reset
    zero_sum = args.zero_sum
    learning_rate = args.learning_rate
    # schedule = ("wolf_adv", 20.0)
    schedule = argument_to_tuple(args.schedule)
    opponent = argument_to_tuple(args.opponent)
    test_every = args.test_every
    save_every = args.save_every
    load = args.load
    load_step = args.load_step
    max_steps = args.max_steps
    test_steps = args.test_steps
    network_width = args.network_width
    network_depth = args.network_depth
    timesteps_per_batch = args.timesteps_per_batch
    iterations_per_round = args.iterations_per_round
    reset_every = args.reset_every
    # other = "1000-test-steps-large-network"

    result_folder = "../result/"
    plot_folder = "../plots/"
    exp_name = args.exp_name or \
        "_".join(["security" + args.other,
                  "belief",
                  # args.other,
                  agent,
                  "beta:{}".format(args.beta),
                  "seed:{}".format(seed),
                  "game:{}-{}-{}-{}".format(n_slots, n_types, n_rounds, "random" if args.random_prior else ":".join(map(str, prior))),
                  "beta1:{}".format(args.beta1),
                  "zs" if zero_sum else "gs",
                  "reset" if reset else "no-reset",
                  "{:.0e}".format(Decimal(learning_rate)),
                  ":".join(list(map(str, schedule))) if type(schedule) == tuple else schedule,
                  # ":".join(list(map(str, train_steps))),
                  ":".join(list(map(str, opponent))) if type(opponent) == tuple else opponent,
                  "test_every:{}".format(test_every),
                  "test_steps:{}".format(test_steps),
                  "reset_every:{}".format(reset_every),
                  "network:{}-{}".format(network_width, network_depth),
                  "train:{}*{}".format(timesteps_per_batch, iterations_per_round)])
    # exp_name = "security_seed:5410_2-2-2_gs_no-reset_5e-6_wolf_adv:20.0_1:1_latest_10"
    exp_dir = os.path.join(result_folder, exp_name)
    plot_dir = os.path.join(plot_folder, exp_name)
    if args.save_dir is not None:
        exp_dir = args.save_dir

    # logger.configure("qweqw.log")
    #train_cnt = np.zeros(shape=(2,2), dtype=np.int32)

    # def train_update_handler(start, actions, rews, infos, done, obs):
    #     # print("ASd")
    #     if actions is not None:
    #         train_cnt[0][actions[0]] += 1
    #         train_cnt[1][actions[1]] += 1

    # test_cnt = np.zeros(shape=(2, 2), dtype=np.int32)
    #
    # def test_update_handler(start, actions, rews, infos, done, obs):
    #     if actions is not None:
    #         test_cnt[0][actions[0]] += 1
    #         test_cnt[1][actions[1]] += 1
    #
    # def show_statistics(cnt):
    #     tot = cnt[0][0] + cnt[0][1]
    #     print("Total iterations: %d" % tot)
    #     print("Agent 0: {:.2%} {:.2%}".format(cnt[0][0] / tot, cnt[0][1] / tot))
    #     print("Agent 1: {:.2%} {:.2%}".format(cnt[1][0] / tot, cnt[1][1] / tot))

    train = True
    # res = {"p": [], "lie_p": []}
    # for ip in range(1, 5):
    #     p = ip / 5.
    #     s = 0.
    #     T = 10
    #     for _ in range(T):
    res = {"episode": [], "assessment": [], "current_assessments": [], "player": []}

    for p in [.5]:
        for _ in range(1):
            env = BeliefSecurityEnv(n_slots=n_slots, n_types=n_types, prior=prior, n_rounds=n_rounds, zero_sum=zero_sum,
                                    seed=seed, export_gambit=n_rounds <= 5 and n_slots <= 2,
                                    random_prior=args.random_prior, vn_load_path=args.vn_load_path, beta=args.beta)
            # env.export_payoff("/home/footoredo/playground/REPEATED_GAME/EXPERIMENTS/PAYOFFSATTvsDEF/%dTarget/inputr-1.000000.csv" % n_slots)
            if train:
                # test_every = 1
                if agent == "ppo":
                    agents = [get_make_ppo_agent(reset_every, timesteps_per_batch, iterations_per_round, beta1=args.beta1, n_rounds=args.n_rounds),
                              get_make_ppo_agent(reset_every, timesteps_per_batch, iterations_per_round, beta1=args.beta1, n_rounds=args.n_rounds)]
                elif agent == "pac":
                    agents = [get_make_pac_agent(timesteps_per_batch, iterations_per_round),
                              get_make_pac_agent(timesteps_per_batch, iterations_per_round)]
                else:
                    raise NotImplementedError
                controller = BeliefController(env, agents)
                train_result, local_results, global_results, train_info = \
                    controller.train(max_steps=max_steps, policy_store_every=None,
                                     test_every=test_every,  test_max_steps=test_steps,
                                     record_assessment=True, reset=reset,
                                     load_state=load, load_path=join_path(exp_dir, "step-{}".format(load_step)),
                                     sub_load_path=args.sub_load_path,
                                     save_every=save_every, save_path=exp_dir, store_results=False)
                import joblib
                joblib.dump({
                    "strategy": train_result["strategy"],
                    "payoff": train_result["payoff"],
                    "distance": train_result["distance"]
                }, join_path_and_check(exp_dir, "part.info"))


else:
    print("fuck")