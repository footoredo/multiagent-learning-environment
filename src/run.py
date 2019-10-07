from env.matrix_env import MatrixEnv
from env.security_env import SecurityEnv
from controller.naive_controller import NaiveController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent_backup import PPOAgent
from agent.mlp_policy import MLPPolicy
import logger
import numpy as np


def make_dummy_agent(observation_space, action_space, handlers):
    return DummyAgent(2)


def make_infant_agent(observation_space, action_space, handlers):
    return InfantAgent(2, handlers)


def make_self_play_agent(observation_space, action_space, handlers):
    return SelfPlayAgent(2, handlers)


ppo_agent_cnt = 0


def make_ppo_agent(observation_space, action_space, handlers):
    def policy(name, ob_space, ac_space):
        return MLPPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=8, num_hid_layers=1)
    global ppo_agent_cnt
    agent = PPOAgent(name="ppo_agent_%d" % ppo_agent_cnt, policy_fn=policy,
                     ob_space=observation_space, ac_space=action_space, handlers=handlers,
                     timesteps_per_actorbatch=1, clip_param=0.2, entcoeff=0.0,
                     optim_epochs=1, optim_stepsize=1e-4,
                     gamma=0.99, lam=0.95, max_episodes=8, schedule="constant")
    ppo_agent_cnt += 1
    return agent


def debugger(infos):
    # print(infos[])
    print(sum(info[0] for info in infos) / len(infos),
          sum(info[1] for info in infos) / len(infos))


if __name__ == "__main__":
    # logger.configure("qweqw.log")

    train_cnt = np.zeros(shape=(2,2), dtype=np.int32)

    def train_update_handler(start, actions, rews, infos, done, obs):
        # print("ASd")
        if actions is not None:
            train_cnt[0][actions[0]] += 1
            train_cnt[1][actions[1]] += 1

    test_cnt = np.zeros(shape=(2, 2), dtype=np.int32)

    def test_update_handler(start, actions, rews, infos, done, obs):
        if actions is not None:
            test_cnt[0][actions[0]] += 1
            test_cnt[1][actions[1]] += 1

    def show_statistics(cnt):
        tot = cnt[0][0] + cnt[0][1]
        print("Total iterations: %d" % tot)
        print("Agent 0: {:.2%} {:.2%}".format(cnt[0][0] / tot, cnt[0][1] / tot))
        print("Agent 1: {:.2%} {:.2%}".format(cnt[1][0] / tot, cnt[1][1] / tot))

    # env = SecurityEnv(n_slots=2, n_types=2, prior=[.7, .3], n_rounds=1)
    env = MatrixEnv()
    controller = NaiveController(env, [make_ppo_agent, make_ppo_agent])
    controller.train(max_steps=5000, policy_store_every=None, test_every=20, test_max_steps=100)
    # show_statistics(train_cnt)
    controller.test(max_steps=100, update_handler=test_update_handler)
    show_statistics(test_cnt)
