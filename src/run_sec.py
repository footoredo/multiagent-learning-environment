from env.matrix_env import MatrixEnv
from env.security_env import SecurityEnv
from controller.naive_controller import NaiveController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent import PPOAgent
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

    parser.add_argument('--seed', type=int)
    parser.add_argument('--agent', type=str, default="ppo")
    parser.add_argument('--other', type=str, default="")
    parser.add_argument('--n-slots', type=int, default=2)
    parser.add_argument('--n-types', type=int, default=2)
    parser.add_argument('--n-rounds', type=int, default=2)
    parser.add_argument('--prior', type=float, nargs='+')
    parser.add_argument('--reset', action="store_true")
    parser.add_argument('--zero-sum', action="store_true")
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--schedule', type=str, nargs="+", default="constant")
    parser.add_argument('--opponent', type=str, nargs="+", default="latest")
    parser.add_argument('--test-every', type=int, default=10)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--load-step', type=int)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--network-width', type=int, default=256)
    parser.add_argument('--network-depth', type=int, default=4)
    parser.add_argument('--test-steps', type=int, default=1000)
    parser.add_argument('--timesteps-per-batch', type=int, default=8)
    parser.add_argument('--iterations-per-round', type=int, default=16)
    parser.add_argument('--exp-name', type=str)

    return parser.parse_args()

# seed = random.randrange(10000)
seed = 5410
# seed = "benchmark"
n_slots = 2
n_types = 2
n_rounds = 5
prior = [.5, .5]
reset = False
zero_sum = False
learning_rate = 5e-4
agent = "ppo"
# schedule = ("wolf_adv", 20.0)
schedule = "constant"
train_steps = [1, 1]
# opponent = "average"
opponent = ("fp", .5)
test_every = 1
save_every = None
max_steps = 4000
network_width = None
network_depth = None
test_steps = None
timesteps_per_batch = None
iterations_per_round = None
other = "1000-test-steps-large-network"

result_folder = "../result/"
exp_name = "_".join(["security",
                     agent,
                     "seed:{}".format(seed),
                     "game:{}-{}-{}-{}".format(n_slots, n_types, n_rounds, ":".join(map(str, prior))),
                     "zs" if zero_sum else "gs",
                     "reset" if reset else "no-reset",
                     "{:.0e}".format(Decimal(learning_rate)),
                     ":".join(list(map(str, schedule))) if type(schedule) == tuple else schedule,
                     # ":".join(list(map(str, train_steps))),
                     ":".join(list(map(str, opponent))) if type(opponent) == tuple else opponent,
                     "every:{}".format(test_every),
                     other])
# exp_name = "security_seed:5410_2-2-2_gs_no-reset_5e-6_wolf_adv:20.0_1:1_latest_10"
exp_dir = os.path.join(result_folder, exp_name)


def get_make_ppo_agent(timesteps_per_actorbatch, max_iterations):
    def make_ppo_agent(observation_space, action_space, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            # return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
            #                  hid_size=256, num_hid_layers=4)
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=network_width, num_hid_layers=network_depth)

        global ppo_agent_cnt
        agent = PPOAgent(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                         ob_space=observation_space, ac_space=action_space, handlers=handlers,
                         timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0,
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
    # other = "1000-test-steps-large-network"

    result_folder = "../result/"
    exp_name = args.exp_name or \
        "_".join(["security" + args.other,
                  agent,
                  "seed:{}".format(seed),
                  "game:{}-{}-{}-{}".format(n_slots, n_types, n_rounds, ":".join(map(str, prior))),
                  "zs" if zero_sum else "gs",
                  "reset" if reset else "no-reset",
                  "{:.0e}".format(Decimal(learning_rate)),
                  ":".join(list(map(str, schedule))) if type(schedule) == tuple else schedule,
                  # ":".join(list(map(str, train_steps))),
                  ":".join(list(map(str, opponent))) if type(opponent) == tuple else opponent,
                  "test_every:{}".format(test_every),
                  "test_steps:{}".format(test_steps),
                  "network:{}-{}".format(network_width, network_depth),
                  "train:{}*{}".format(timesteps_per_batch, iterations_per_round)])
    # exp_name = "security_seed:5410_2-2-2_gs_no-reset_5e-6_wolf_adv:20.0_1:1_latest_10"
    exp_dir = os.path.join(result_folder, exp_name)

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
    res = {"episode": [], "assessment": [], "player": []}

    for p in [.5]:
        for _ in range(1):
            env = SecurityEnv(n_slots=n_slots, n_types=n_types, prior=prior, n_rounds=n_rounds, zero_sum=zero_sum,
                              seed=seed, export_gambit=n_rounds <= 5 and n_slots <= 2)
            env.export_payoff("/home/footoredo/playground/REPEATED_GAME/EXPERIMENTS/PAYOFFSATTvsDEF/%dTarget/inputr-1.000000.csv" % n_slots)
            if train:
                # test_every = 1
                if agent == "ppo":
                    agents = [get_make_ppo_agent(timesteps_per_batch, iterations_per_round),
                              get_make_ppo_agent(timesteps_per_batch, iterations_per_round)]
                elif agent == "pac":
                    agents = [get_make_pac_agent(timesteps_per_batch, iterations_per_round),
                              get_make_pac_agent(timesteps_per_batch, iterations_per_round)]
                else:
                    raise NotImplementedError
                controller = NaiveController(env, agents)
                train_result, local_results, train_info = \
                    controller.train(max_steps=max_steps, policy_store_every=None,
                                     test_every=test_every,  test_max_steps=test_steps,
                                     record_assessment=True, train_steps=train_steps, reset=reset,
                                     load_state=load, load_path=join_path(exp_dir, "step-{}".format(load_step)),
                                     save_every=save_every, save_path=exp_dir, store_results=False)
                env.export_settings(join_path_and_check(exp_dir, "env_settings.obj"))
                assessments = train_result["assessments"]
                joblib.dump(train_result["final_assessment"], join_path_and_check(exp_dir, "final_assessment.obj"))
                joblib.dump(local_results, join_path_and_check(exp_dir, "local_results.obj"))
                joblib.dump(train_info, join_path_and_check(exp_dir, "train_info.obj"))
                # joblib.dump(train_result["local_"], join_path_and_check(exp_dir, "final_assessment.obj"))
                print(assessments)
                print(train_result["random_assessment"])
                for i in range(test_every, max_steps, test_every):
                    res["episode"].append(i)
                    res["assessment"].append(assessments[i // test_every - 1][0][0])
                    res["player"].append("attacker")

                    res["episode"].append(i)
                    res["assessment"].append(assessments[i // test_every - 1][1][0])
                    res["player"].append("defender")

                    res["episode"].append(i)
                    res["assessment"].append(assessments[i // test_every - 1][0][1])
                    res["player"].append("attacker PBNE")

                    res["episode"].append(i)
                    res["assessment"].append(assessments[i // test_every - 1][1][1])
                    res["player"].append("defender PBNE")
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
    sns.lineplot(x="episode", y="assessment", hue="player", data=df)
    plt.savefig(join_path_and_check(exp_dir, "result.png"), dpi=800)
    plt.show()
else:
    print("fuck")