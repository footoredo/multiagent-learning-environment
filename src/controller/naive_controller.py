from env.base_env import BaseEnv
from env.reduced_env import ReducedEnv
from env.monitor_env import MonitorEnv
from controller.base_controller import BaseController
from agent.base_agent import BaseAgent
from monitor.statistics import Statistics
from agent.policy import MixedPolicy
import random
import time
import numpy as np
import joblib
from common.path_utils import *


class NaiveController(BaseController):
    def __init__(self, env: BaseEnv, agent_fns):
        super().__init__(env)
        self.agent_fns = agent_fns
        self.step = None
        self.policy_store_every = None
        self.policy_pool_size = None
        self.test_cnt = np.zeros(shape=(2, 2), dtype=np.int32)
        self.statistics = None
        self.records = None
        self.push_list = None
        self.latest_policy = None
        self.avg_policy = None
        self.policy_pool = None

    def get_push_handler(self, i):
        def push(policy, avg_policy):
            # self.push_list.append((i, policy))
            self._push_policy(i, policy, avg_policy)
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
            k = None
            if type(version) == tuple:
                version, k = version
            if version == "average":
                # fixed_policies = [self.statistics.get_avg_policy(j) for j in range(self.num_agents) if j != i]
                fixed_policies = [MixedPolicy([self.statistics.get_avg_policy(j),
                                               self.latest_policy[j]],
                                              [1 - k, k]
                                              ) for j in range(self.num_agents) if j != i]
            elif version == "fp":
                fixed_policies = []
                for j in range(self.num_agents):
                    if j != i:
                        if random.random() < k:
                            fixed_policies.append(self.latest_policy[j])
                        else:
                            fixed_policies.append(self.statistics.get_avg_policy(j))
            elif version == "latest":
                fixed_policies = [self.latest_policy[j] for j in range(self.num_agents) if j != i]
            else:
                raise NotImplementedError

            return ReducedEnv(self.env,
                              fixed_indices=[j for j in range(self.num_agents) if j != i],
                              fixed_policies=fixed_policies,
                              allow_single=True)
        return pull

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _push_policy(self, i, policy, avg_policy):
        if self.policy_store_every is not None:
            raise NotImplementedError
        else:
            self.latest_policy[i] = policy
            self.avg_policy[i] = avg_policy
        # if self.policy_store_every is not None and self.step > 0 and self.step % self.policy_store_every == 0:
        #     self.policies[i].append(policy)
        # else:
        #     self.policies[i][-1] = policy

    def save(self, save_path):
        for i, agent in enumerate(self.agents):
            agent.save(join_path(save_path, "agent-{}".format(i)))
        self.statistics.save(join_path_and_check(save_path, "statistics.obj"))
        joblib.dump((self.step, self.records), join_path_and_check(save_path, "records.obj"), compress=3)

    def load(self, load_path):
        for i, agent in enumerate(self.agents):
            agent.load(join_path(load_path, "agent-{}".format(i)))
        self.statistics.load(join_path(load_path, "statistics.obj"))
        self.step, self.records = joblib.load(join_path(load_path, "records.obj"))

    def _train(self, max_steps=10000, policy_store_every=100, policy_pool_size=50, test_every=100,
               show_every=None, test_max_steps=100, record_assessment=False, train_steps=None, reset=False,
               sec_prob=False, save_every=None, save_path=None, load_state=None, load_path=None, store_results=False):
        if train_steps is None:
            train_steps = [1 for _ in range(self.env.num_agents)]
        self.policy_store_every = policy_store_every
        self.policy_pool_size = policy_pool_size
        # env = self.env if update_handler is None else MonitorEnv(self.env, update_handler)
        env = self.env
        self.agents = [agent_fn(observation_space=env.get_observation_space(i),
                                action_space=env.get_action_space(i),
                                handlers=self.get_handlers(i))
                       for i, agent_fn in enumerate(self.agent_fns)]
        for agent in self.agents:
            assert isinstance(agent, BaseAgent)
        self.agent_configs = [agent.get_config() for agent in self.agents]

        self.statistics = Statistics(self.env) if test_every is not None else None

        self.records = {
            "local_results": [],
            "global_results": [],
            "assessments": []
        }

        self.step = 0

        if load_state:
            self.load(load_path)

        if store_results:
            local_results = self.records["local_results"]
            global_results = self.records["global_results"]
        else:
            local_results = []
            global_results = []
        assessments = self.records["assessments"]
        self.latest_policy = [agent.get_initial_policy() for agent in self.agents]
        self.avg_policy = [None for agent in self.agents]
        self.policy_pool = [[] for _ in self.agents]

        def check_every(every):
            return every is not None and self.step > 0 and self.step % every == 0

        self.push_list = []

        train_info = [[] for _ in range(self.num_agents)]

        last_time = time.time()
        while self.step < max_steps:
            for i, policy in self.push_list:
                self._push_policy(i, policy)
            for i, agent in enumerate(self.agents):
                assert(train_steps[i] == 1)
                for _ in range(train_steps[i]):
                    if sec_prob:
                        env.update_attacker_policy(self.get_policy_with_version(self.policies[0], version="latest"))
                    train_info[i].append(agent.train(i, self.statistics, self.step / max_steps, self.step))

            self.step += 1
            self.env.update_attacker_average_policy(self.latest_policy[0].strategy_fn)
            self.env.update_defender_average_policy(self.latest_policy[1].strategy_fn)

            if reset and self.step / max_steps > .3:
                self.statistics.reset()
                print("RESET!")
                reset = False
            if check_every(test_every):
                local_result, global_result = self.run_test(test_max_steps)
                if store_results:
                    local_results.append(local_result)
                    global_results.append(global_result)
                if record_assessment:
                    # rews = self.run_benchmark(1000)
                    print("current")
                    assessment = self.env.assess_strategies([self.latest_policy[i].strategy_fn
                                                             for i in range(self.num_agents)])
                    print("avg network")
                    assessment = self.env.assess_strategies([self.avg_policy[i].strategy_fn
                                                             for i in range(self.num_agents)])
                    print("avg")
                    assessment = self.env.assess_strategies([self.statistics.get_avg_strategy(i)
                                                             for i in range(self.num_agents)])
                    print("true avg")
                    assessment = self.env.assess_strategies([self.env.get_attacker_average_policy(),
                                                             self.env.get_defender_average_policy()])
                    # for i in range(self.num_agents):
                    #     assessment.append(self.env.assess_strategy(i, self.statistics.get_avg_strategy(i)))
                    # exp[0] += 0.633
                    # exp[1] += 2.387
                    assessments.append(assessment)
                    print("Current assessment:", assessment)
                    # self.run_benchmark()

                now_time = time.time()
                print("\n### Step %d / %d" % (self.step, max_steps), now_time - last_time)
                last_time = now_time

            if check_every(show_every):
                self.show()

            if check_every(save_every):
                self.save(join_path_and_check(save_path, "step-{}".format(self.step)))

        if test_every is not None:
            self.statistics.show_statistics()

        if record_assessment:
            final_assessment = self.env.assess_strategies([self.statistics.get_avg_strategy(i, trim_th=0.)
                                                           for i in range(self.num_agents)], verbose=True)
            self.records["final_assessment"] = final_assessment
            random_statistics = Statistics(self.env)
            random_assessment = self.env.assess_strategies([random_statistics.get_avg_strategy(i)
                                                           for i in range(self.num_agents)], verbose=False)
            self.records["random_assessment"] = random_assessment

        # self.run_benchmark(10000)
        return self.records, local_results, train_info

    def run_benchmark(self, max_steps=500):
        statistics = Statistics(self.env)
        benchmark_env = MonitorEnv(self.env, update_handlers=statistics.get_update_handler())
        final_policies = [self.statistics.get_avg_policy(i) for i in range(len(self.agents))]
        benchmark_env = ReducedEnv(benchmark_env,
                                   fixed_indices=range(self.num_agents),
                                   fixed_policies=final_policies)
        # print("ASd")
        benchmark_env.reset()
        for step in range(max_steps):
            _, _, _, done = benchmark_env.step([])
            if done:
                benchmark_env.reset()
        avg_rews_per_player = statistics.get_avg_rews_per_player()
        statistics.show_statistics()
        return avg_rews_per_player

    @staticmethod
    def show_statistics(cnt):
        tot = cnt[0][0] + cnt[0][1]
        print("Total iterations: %d" % tot)
        print("Agent 0: {:.2%} {:.2%}".format(cnt[0][0] / tot, cnt[0][1] / tot))
        print("Agent 1: {:.2%} {:.2%}".format(cnt[1][0] / tot, cnt[1][1] / tot))

    def run_test(self, max_steps):
        local_statistics = Statistics(self.env)

        def double_update_handler(last_obs, start, actions, rews, infos, done, obs, history):
            local_statistics.get_update_handler()(last_obs, start, actions, rews, infos, done, obs, history)
            self.statistics.get_update_handler()(last_obs, start, actions, rews, infos, done, obs, history)

        self._test(max_steps, update_handler=double_update_handler)
        # local_statistics.show_statistics()
        # self.statistics.show_statistics()
        return local_statistics.export_statistics(), self.statistics.export_statistics()
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
            _, _, _, done, _, _ = test_env.step([], [])
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