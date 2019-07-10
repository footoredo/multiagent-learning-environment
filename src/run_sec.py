from env.matrix_env import MatrixEnv
from env.security_env import SecurityEnv
from controller.naive_controller import NaiveController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent import PPOAgent
from agent.mlp_policy import MLPPolicy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from decimal import Decimal
from common.path_utils import *
import joblib


def make_dummy_agent(observation_space, action_space, handlers):
    return DummyAgent(2)


def make_infant_agent(observation_space, action_space, handlers):
    return InfantAgent(2, handlers)


def make_self_play_agent(observation_space, action_space, handlers):
    return SelfPlayAgent(2, handlers)


ppo_agent_cnt = 0

# seed = random.randrange(10000)
seed = 5410
# seed = "benchmark"
n_slots = 2
n_types = 2
n_rounds = 10
reset = False
zero_sum = False
learning_rate = 5e-6
schedule = ("wolf_adv", 20.0)
# schedule = "constant"
train_steps = [1, 1]
opponent = "latest"
test_every = 10
max_steps = 10000

result_folder = "../result/"
exp_name = "_".join(["security",
                     "seed:{}".format(seed),
                     "{}-{}-{}".format(n_slots, n_types, n_rounds),
                     "zs" if zero_sum else "gs",
                     "reset" if reset else "no-reset",
                     "{:.0e}".format(Decimal(learning_rate)),
                     ":".join(list(map(str, schedule))),
                     ":".join(list(map(str, train_steps))),
                     opponent,
                     "{}".format(test_every)])
exp_dir = os.path.join(result_folder, exp_name)


def get_make_ppo_agent(timesteps_per_actorbatch, max_episodes):
    def make_ppo_agent(observation_space, action_space, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=16, num_hid_layers=2)

        global ppo_agent_cnt
        agent = PPOAgent(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                         ob_space=observation_space, ac_space=action_space, handlers=handlers,
                         timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0.0,
                         optim_epochs=1, optim_stepsize=learning_rate,
                         gamma=0.99, lam=0.95, max_episodes=max_episodes,
                         schedule=schedule, opponent=opponent)
        ppo_agent_cnt += 1
        return agent
    return make_ppo_agent


def debugger(infos):
    # print(infos[])
    print(sum(info[0] for info in infos) / len(infos),
          sum(info[1] for info in infos) / len(infos))


if __name__ == "__main__":
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
            env = SecurityEnv(n_slots=n_slots, n_types=n_types, prior=[p, 1. - p], n_rounds=n_rounds, zero_sum=zero_sum, seed=seed)
            env.export_payoff("/home/footoredo/playground/REPEATED_GAME/EXPERIMENTS/PAYOFFSATTvsDEF/%dTarget/inputr-1.000000.csv" % n_slots)
            env.export_settings("../result/setting.pkl")
            if train:
                # test_every = 1
                controller = NaiveController(env, [get_make_ppo_agent(8, 16), get_make_ppo_agent(8, 16)])
                train_result = controller.train(max_steps=max_steps, policy_store_every=None,
                                                test_every=test_every,  test_max_steps=100,
                                                record_assessment=True, train_steps=train_steps, reset=reset,
                                                load_state=False, load_path=join_path(exp_dir, "step-10"),
                                                save_every=1000, save_path=exp_dir)
                assessments = train_result["assessments"]
                print(assessments)
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
                lie_p = env.get_lie_prob()
                print(p, lie_p)
                s += lie_p
                # res["p"].append(p)
                # res["lie_p"].append(lie_p)
        # res["p"].append(p)
        # res["lie_p"].append(s / T)

    joblib.dump(res, join_path_and_check(exp_dir, "result.obj"))
    df = pd.DataFrame(data=res)
    sns.set()
    sns.lineplot(x="episode", y="assessment", hue="player", data=df)
    plt.savefig(join_path_and_check(exp_dir, "result.png"))
    plt.show()
