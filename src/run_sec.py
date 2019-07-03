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
                         optim_epochs=1, optim_stepsize=5e-6,
                         gamma=0.99, lam=0.95, max_episodes=max_episodes, schedule="constant")
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
    res = {"episode": [], "exploitability": []}
    for p in [.3]:
        for _ in range(1):
            env = SecurityEnv(n_slots=2, n_types=2, prior=[p, 1. - p], n_rounds=2)
            if train:
                max_steps = 5000
                test_every = 10
                controller = NaiveController(env, [get_make_ppo_agent(8, 16), get_make_ppo_agent(8, 16)])
                _, _, exp = controller.train(max_steps=max_steps, policy_store_every=None, test_every=test_every, test_max_steps=500)
                for i in range(test_every, max_steps, test_every):
                    res["episode"].append(i)
                    res["exploitability"].append(exp[i // test_every - 1])
            else:
                lie_p = env.get_lie_prob()
                print(p, lie_p)
                s += lie_p
                # res["p"].append(p)
                # res["lie_p"].append(lie_p)
        # res["p"].append(p)
        # res["lie_p"].append(s / T)

    df = pd.DataFrame(data=res)
    sns.set()
    sns.relplot(x="episode", y="exploitability", data=df)
    plt.show()
