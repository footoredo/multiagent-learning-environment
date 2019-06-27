from env.base_env import BaseEnv
from env.reduced_env import ReducedEnv
from env.monitor_env import MonitorEnv
from controller.base_controller import BaseController
from agent.base_agent import BaseAgent
from monitor.statistics import Statistics
import random
import time
import numpy as np


class NaiveController(BaseController):
    def __init__(self, env: BaseEnv, agent_fns):
        super().__init__(env)
        self.agent_fns = agent_fns
        self.step = None
        self.policy_store_every = None
        self.test_cnt = np.zeros(shape=(2, 2), dtype=np.int32)
        self.statistics = None

    def get_push_handler(self, i):
        def push(policy):
            self._push_policy(i, policy)
        return push

    @staticmethod
    def get_policy_with_version(policies, version):
        if version == "latest":
            return policies[-1]
        elif version == "random":
            return random.choice(policies)
        elif type(version) == int:
            return policies[version]
        else:
            raise NotImplementedError

    def get_pull_handler(self, i):
        def pull(version="latest"):
            # if version == "latest":
            #     version = -1
            # if type(version) == int:
            #     version -= i
            if version == "average":
                fixed_policies = [self.statistics.get_avg_policy(j) for j in range(self.num_agents) if j != i]
            else:
                fixed_policies = [self.get_policy_with_version(
                    self.policies[j], version=version
                ) for j in range(self.num_agents) if j != i]

            return ReducedEnv(self.env,
                              fixed_indices=[j for j in range(self.num_agents) if j != i],
                              fixed_policies=fixed_policies,
                              allow_single=True)
        return pull

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _push_policy(self, i, policy):
        if self.policy_store_every is not None and self.step > 0 and self.step % self.policy_store_every == 0:
            self.policies[i].append(policy)
        else:
            self.policies[i][-1] = policy

    def _train(self, max_steps=10000, policy_store_every=100, test_every=100,
               show_every=None, test_max_steps=100):
        self.policy_store_every = policy_store_every
        # env = self.env if update_handler is None else MonitorEnv(self.env, update_handler)
        env = self.env
        results = []
        self.agents = [agent_fn(observation_space=env.get_observation_space(i),
                                action_space=env.get_action_space(i),
                                handlers=self.get_handlers(i))
                       for i, agent_fn in enumerate(self.agent_fns)]
        for agent in self.agents:
            assert isinstance(agent, BaseAgent)
        self.agent_configs = [agent.get_config() for agent in self.agents]
        self.policies = [[agent.get_initial_policy()] for agent in self.agents]

        last_time = time.time()

        self.statistics = Statistics(self.env) if test_every is not None else None

        for self.step in range(max_steps):
            if test_every is not None and self.step % test_every == 0 and self.step > 0:
                now_time = time.time()
                print("\n### Step %d / %d" % (self.step, max_steps), now_time - last_time)
                last_time = now_time
                results.append(self.run_test(test_max_steps))
            if show_every is not None and self.step % show_every == 0 and self.step > 0:
                self.show()
            for i, agent in enumerate(self.agents):
                for _ in range(1):
                    agent.train(i, self.statistics, self.step / max_steps)

        if test_every is not None:
            self.statistics.show_statistics()

        return results

    @staticmethod
    def show_statistics(cnt):
        tot = cnt[0][0] + cnt[0][1]
        print("Total iterations: %d" % tot)
        print("Agent 0: {:.2%} {:.2%}".format(cnt[0][0] / tot, cnt[0][1] / tot))
        print("Agent 1: {:.2%} {:.2%}".format(cnt[1][0] / tot, cnt[1][1] / tot))

    def run_test(self, max_steps):
        local_statistics = Statistics(self.env)

        def double_update_handler(last_obs, start, actions, rews, infos, done, obs):
            local_statistics.get_update_handler()(last_obs, start, actions, rews, infos, done, obs)
            self.statistics.get_update_handler()(last_obs, start, actions, rews, infos, done, obs)

        self._test(max_steps, update_handler=double_update_handler)
        local_statistics.show_statistics()
        self.statistics.show_statistics()
        return local_statistics.export_statistics()
        # return self.statistics.export_statistics()

    def run_test_old(self, max_steps):
        test_cnt = np.zeros(shape=(2, 2), dtype=np.int32)

        def test_update_handler(start, actions, rews, infos, done, obs):
            if actions is not None:
                test_cnt[0][actions[0]] += 1
                test_cnt[1][actions[1]] += 1
                self.test_cnt[0][actions[0]] += 1
                self.test_cnt[1][actions[1]] += 1

        self._test(max_steps, update_handler=test_update_handler)
        self.show_statistics(test_cnt)

    def test(self, *args, **kwargs):
        self._test(*args, **kwargs)

    def show(self):
        show_env = self.env
        policies = [agent.get_final_policy() for agent in self.agents]
        show_env = ReducedEnv(show_env,
                              fixed_indices=range(self.num_agents),
                              fixed_policies=policies)
        show_env.reset(debug=True)
        while True:
            _, _, _, done = show_env.step([])
            if done:
                break

    def _test(self, max_steps=10000, update_handler=None):
        test_env = self.env if update_handler is None else MonitorEnv(self.env, update_handler)
        final_policies = [agent.get_final_policy() for agent in self.agents]
        test_env = ReducedEnv(test_env,
                              fixed_indices=range(self.num_agents),
                              fixed_policies=final_policies)
        # print("ASd")
        test_env.reset()
        for step in range(max_steps):
            _, _, _, done = test_env.step([])
            if done:
                test_env.reset()

    @staticmethod
    def expand_info(seg):
        for key, _ in seg["info"][0].items():
            seg[key] = []
        for info in seg["info"]:
            for key, item in info.items():
                seg[key].append(item)

    def _train_old(self, max_steps=10000, horizon=100, debugger=None):
        obs = self.env.reset()
        obs_seq = [[None for _ in range(horizon)] for _ in range(self.num_agents)]
        acs_seq = [[None for _ in range(horizon)] for _ in range(self.num_agents)]
        rews_seq = [[None for _ in range(horizon)] for _ in range(self.num_agents)]
        news_seq = [[False for _ in range(horizon)] for _ in range(self.num_agents)]
        infos_seq = [None for _ in range(horizon)]
        actor_info_seq = [[None for _ in range(horizon)] for _ in range(self.num_agents)]

        new = True

        for step in range(max_steps):
            if step > 0 and step % horizon == 0:
                self.policies = [self.agents[i].update(self.expand_info({
                    "ob": obs_seq[i],
                    "ac": acs_seq[i],
                    "rew": rews_seq[i],
                    "new": news_seq[i],
                    "info": actor_info_seq[i]
                })) or self.policies[i] for i in range(self.num_agents)]
                debugger(infos_seq)

            acis = [policy.act(ob) for policy, ob in zip(self.policies, obs)]  # ac, info_dict
            for i in range(self.num_agents):
                news_seq[i][step % horizon] = new
                obs_seq[i][step % horizon] = obs[i]
                acs_seq[i][step % horizon] = acis[i][0]
                actor_info_seq[i][step % horizon] = acis[i][1]

            new = False

            obs, rews, infos, done = self.env.step(acs)
            for i in range(self.num_agents):
                rews_seq[i][step % horizon] = rews[i]
            infos_seq[step % horizon] = infos

            if done:
                obs = self.env.reset()
                new = True
