from env.matrix_env import MatrixEnv
from env.belief_deceive_env import DeceiveEnv
from controller.belief_controller_pw import NaiveController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent import PPOAgent
from agent.pac_agent import PACAgent
from agent.mlp_policy import MLPPolicy
# from agent.recurrent_policy import RecurrentPolicy
from agent.recurrent_policy_tpred import RecurrentPolicy
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

    parser.add_argument('--agent', type=str, default="ppo")
    parser.add_argument('--n-targets', type=int, default=2)
    parser.add_argument('--n-rounds', type=int, default=2)
    parser.add_argument('--steps-per-round', type=int, default=5)
    parser.add_argument('--prior', type=float, nargs='+')
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--test-every', type=int, default=10)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--random-prior', action="store_true")
    parser.add_argument('--load-step', type=int)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--network-width', type=int, default=256)
    parser.add_argument('--network-depth', type=int, default=4)
    parser.add_argument('--timesteps-per-batch', type=int, default=8)
    parser.add_argument('--iterations-per-round', type=int, default=16)
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--other', type=str, default='')

    return parser.parse_args()


def get_make_ppo_agent(timesteps_per_actorbatch, max_iterations):
    def make_ppo_agent(observation_space, action_space, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            # init_ob_space, info_ob_space = ob_space
            # return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
            #                  hid_size=256, num_hid_layers=4)
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=network_width, num_hid_layers=network_depth)

        global ppo_agent_cnt
        agent = PPOAgent(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                         ob_space=observation_space,
                         ac_space=action_space, handlers=handlers,
                         timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0,
                         optim_epochs=1, optim_stepsize=learning_rate, beta1=0.9,
                         gamma=0.99, lam=0.95, max_iters=max_iterations,
                         schedule="constant", opponent="latest")
        ppo_agent_cnt += 1
        return agent
    return make_ppo_agent


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

    # seed = "benchmark"
    agent = args.agent
    n_targets = args.n_targets
    n_rounds = args.n_rounds
    steps_per_round = args.steps_per_round
    prior = args.prior
    learning_rate = args.learning_rate
    # schedule = ("wolf_adv", 20.0)
    test_every = args.test_every
    save_every = args.save_every
    load = args.load
    load_step = args.load_step
    max_steps = args.max_steps
    network_width = args.network_width
    network_depth = args.network_depth
    timesteps_per_batch = args.timesteps_per_batch
    iterations_per_round = args.iterations_per_round
    # other = "1000-test-steps-large-network"

    result_folder = "../result/"
    plot_folder = "../plots/"
    exp_name = args.exp_name or \
        "_".join(["deceive" + str(args.other),
                  "recurrent",
                  agent,
                  "game:{}-{}-{}-{}".format(n_targets, n_rounds, steps_per_round, ":".join(map(str, prior))),
                  "{:.0e}".format(Decimal(learning_rate)),
                  "test_every:{}".format(test_every),
                  "network:{}-{}".format(network_width, network_depth),
                  "train:{}*{}".format(timesteps_per_batch, iterations_per_round)])
    # exp_name = "security_seed:5410_2-2-2_gs_no-reset_5e-6_wolf_adv:20.0_1:1_latest_10"
    exp_dir = os.path.join(result_folder, exp_name)
    plot_dir = os.path.join(plot_folder, exp_name)

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
    res = {"episode": [], "current_assessments": [], "player": []}

    for p in [.5]:
        for _ in range(1):
            env = DeceiveEnv(n_targets=n_targets, n_rounds=n_rounds, steps_per_round=steps_per_round, prior=prior,
                             random_prior=args.random_prior)
            # env.export_payoff("/home/footoredo/playground/REPEATED_GAME/EXPERIMENTS/PAYOFFSATTvsDEF/%dTarget/inputr-1.000000.csv" % n_slots)
            if train:
                # test_every = 1
                if agent == "ppo":
                    agents = [get_make_ppo_agent(timesteps_per_batch, iterations_per_round),
                              get_make_ppo_agent(timesteps_per_batch, iterations_per_round)]
                else:
                    raise NotImplementedError
                controller = NaiveController(env, agents)
                train_result, local_results, train_info = \
                    controller.train(max_steps=max_steps, policy_store_every=None,
                                     test_every=test_every,  test_max_steps=0,
                                     record_assessment=True, reset=False,
                                     load_state=load, load_path=join_path(exp_dir, "step-{}".format(load_step)),
                                     save_every=save_every, save_path=exp_dir, store_results=False)
                assessments = train_result["assessments"]
                current_assessments = train_result["current_assessments"]
                joblib.dump(train_result["final_assessment"], join_path_and_check(exp_dir, "final_assessment.obj"))
                joblib.dump(local_results, join_path_and_check(exp_dir, "local_results.obj"))
                joblib.dump(train_info, join_path_and_check(exp_dir, "train_info.obj"))
                # joblib.dump(train_result["local_"], join_path_and_check(exp_dir, "final_assessment.obj"))
                # print(assessments)
                # print(train_result["random_assessment"])
                for i in range(test_every, max_steps + 1, test_every):
                    res["episode"].append(i)
                    res["current_assessments"].append(current_assessments[i // test_every - 1][0])
                    res["player"].append("attacker")

                    res["episode"].append(i)
                    res["current_assessments"].append(current_assessments[i // test_every - 1][1])
                    res["player"].append("defender")
            else:
                pass
                # lie_p = env.get_lie_prob()
                # print(p, lie_p)
                # s += lie_p
                # res["p"].append(p)
                # res["lie_p"].append(lie_p)
        # res["p"].append(p)
        # res["lie_p"].append(s / T)

    print(exp_name)
    joblib.dump(res, join_path_and_check(exp_dir, "result.obj"))
    df = pd.DataFrame(data=res)
    sns.set()
    # sns.lineplot(x="episode", y="assessment", hue="player", data=df)
    # plt.savefig(join_path_and_check(plot_dir, "result.png"), dpi=800)
    # plt.show()
    sns.lineplot(x="episode", y="current_assessments", hue="player", data=df)
    plt.savefig(join_path_and_check(plot_dir, "current_result.png"), dpi=800)
    plt.show()
else:
    print("fuck")