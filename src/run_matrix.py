from env.matrix_env import MatrixEnv
from env.security_env import SecurityEnv
from controller.naive_controller import NaiveController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent_backup import PPOAgent
from agent.mlp_policy import MLPPolicy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logger
import numpy as np
import pickle
from decimal import Decimal


def make_dummy_agent(observation_space, action_space, handlers):
    return DummyAgent(2)


def make_infant_agent(observation_space, action_space, handlers):
    return InfantAgent(2, handlers)


def make_self_play_agent(observation_space, action_space, handlers):
    return SelfPlayAgent(2, handlers)


ppo_agent_cnt = 0

lr_schedule = ("wolf_adv", 20.0)
init_lr = 5e-6
opponent = "latest"
experiment_name = "_".join(["matrix",
                            ":".join(list(map(str, lr_schedule))),
                            "{:.0e}".format(Decimal(init_lr)),
                            opponent
                            ])


def get_make_ppo_agent(timesteps_per_actorbatch, max_episodes):
    def make_ppo_agent(observation_space, action_space, handlers):
        def policy(name, agent_name, ob_space, ac_space):
            return MLPPolicy(name=name, agent_name=agent_name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=8, num_hid_layers=1)

        global ppo_agent_cnt
        agent = PPOAgent(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                         ob_space=observation_space, ac_space=action_space, handlers=handlers,
                         timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0.0,
                         optim_epochs=8, optim_stepsize=init_lr,
                         gamma=0.99, lam=0.95, max_episodes=max_episodes, schedule=lr_schedule, opponent=opponent)
        ppo_agent_cnt += 1
        return agent
    return make_ppo_agent


def debugger(infos):
    # print(infos[])
    print(sum(info[0] for info in infos) / len(infos),
          sum(info[1] for info in infos) / len(infos))


def update_res(result, res):
    a0p0 = result[0][eob][0]
    a0p1 = result[0][eob][1]
    a1p0 = result[1][eob][0]
    a1p1 = result[1][eob][1]

    regret_0 = max(calc_u(0., a1p0), calc_u(1., a1p0)) - calc_u(a0p0, a1p0)
    regret_1 = max(-calc_u(a0p0, 0.), -calc_u(a0p0, 1.)) + calc_u(a0p0, a1p0)

    res["episode"].append(i)
    res["regret"].append(regret_0)
    res["prob"].append(a0p0)
    res["player"].append("Player 1")

    res["episode"].append(i)
    res["regret"].append(regret_1)
    res["prob"].append(a1p0)
    res["player"].append("Player 2")


if __name__ == "__main__":
    print(experiment_name)
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
    lres = dict()
    lres["episode"] = []
    lres["regret"] = []
    lres["prob"] = []
    lres["player"] = []

    gres = dict()
    gres["episode"] = []
    gres["regret"] = []
    gres["prob"] = []
    gres["player"] = []


    def calc_u(xa0p0, xa1p0):
        payoff = [[0., -2.], [-7., 1.]]
        xa0p1 = 1 - xa0p0
        xa1p1 = 1 - xa1p0
        u = payoff[0][0] * xa0p0 * xa1p0 + payoff[0][1] * xa0p0 * xa1p1 + \
            payoff[1][0] * xa0p1 * xa1p0 + payoff[1][1] * xa0p1 * xa1p1
        return u

    for _ in range(1):
        env = MatrixEnv()
        if train:
            n_ep = 10000
            t_ev = 20
            controller = NaiveController(env, [get_make_ppo_agent(1, 8), get_make_ppo_agent(1, 8)])
            local_results, global_results, exp, _ = \
                controller.train(max_steps=n_ep, policy_store_every=None, test_every=t_ev, test_max_steps=100,
                                 record_exploitability=True)
            print(exp)
            # eob = env.get_ob_encoders()[0](np.zeros(1))
            # for i in range(t_ev, n_ep, t_ev):
            #     update_res(local_results[i // t_ev - 1], lres)
            #     update_res(global_results[i // t_ev - 1], gres)

                # res["episode"].append(i)
                # res["prob"].append(results[i // t_ev - 1][0][eob][0])
                # res["player"].append("Player 1")
                # res["episode"].append(i)
                # res["prob"].append(4. / 5.)
                # res["player"].append("Player 1 Equilibrium")
                # res["episode"].append(i)
                # res["prob"].append(results[i // t_ev - 1][1][eob][0])
                # res["player"].append("Player 2")
                # res["episode"].append(i)
                # res["prob"].append(3. / 10.)
                # res["player"].append("Player 2 Equilibrium")
                # res["p0a0"].append(r[0][eob][0])
                # res["p0a1"].append(r[0][eob][1])
                # res["p1a0"].append(r[1][eob][0])
                # res["p1a1"].append(r[1][eob][1])

            # print(len(res["p0a0"]))

        else:
            pass
            # lie_p = env.get_lie_prob()
            # print(p, lie_p)
            # s += lie_p
            # res["p"].append(p)
            # res["lie_p"].append(lie_p)
        # res["p"].append(p)
        # res["lie_p"].append(s / T)

    pickle.dump(lres, open(experiment_name + "_local.obj", "wb"))
    pickle.dump(gres, open(experiment_name + "_global.obj", "wb"))
    df = pd.DataFrame(data=gres)
    sns.set()
    sns.relplot(x="episode", y="regret", hue="player", kind="line", data=df)
    plt.show()
