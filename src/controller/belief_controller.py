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
import copy
from common.path_utils import *


class BeliefController(BaseController):
    def __init__(self, env: BaseEnv, agent_fns):
        super().__init__(env)
        self.agent_fns = agent_fns
        self.step = None
        self.policy_store_every = None
        self.policy_pool_size = None
        self.test_cnt = np.zeros(shape=(2, 2), dtype=np.int32)
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
                raise NotImplementedError
            elif version == "fp":
                raise NotImplementedError
            elif version == "latest":
                fixed_policies = [self.latest_policy[j] for j in range(self.num_agents) if j != i]
            else:
                raise NotImplementedError

            return ReducedEnv(self.env if i == 1 else self.atk_env,
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
        # self.statistics.save(join_path_and_check(save_path, "statistics.obj"))
        joblib.dump((self.step, self.records), join_path_and_check(save_path, "records.obj"), compress=3)

    def load(self, load_path):
        for i, agent in enumerate(self.agents):
            agent.load(join_path(load_path, "agent-{}".format(i)))
        # self.statistics.load(join_path(load_path, "statistics.obj"))
        self.step, self.records = joblib.load(join_path(load_path, "records.obj"))

    def _train(self, max_steps=10000, policy_store_every=100, policy_pool_size=50, test_every=100,
               show_every=None, test_max_steps=100, record_assessment=False, train_steps=None, reset=False,
               sec_prob=False, save_every=None, save_path=None, load_state=None, load_path=None, store_results=False,
               sub_load_path=None):
        if train_steps is None:
            train_steps = [1 for _ in range(self.env.num_agents)]
        train_steps = [1, 1]
        self.policy_store_every = policy_store_every
        self.policy_pool_size = policy_pool_size
        # env = self.env if update_handler is None else MonitorEnv(self.env, update_handler)
        env = self.env
        self.atk_env = self.env.get_atk_env()
        self.agents = [agent_fn(observation_space=env.get_observation_space(i),
                                action_space=env.get_action_space(i),
                                steps_per_round=env.n_rounds,
                                handlers=self.get_handlers(i))
                       for i, agent_fn in enumerate(self.agent_fns)]

        self.agent_configs = [agent.get_config() for agent in self.agents]

        # self.statistics = Statistics(self.env) if test_every is not None else None

        self.records = {
            "local_results": [],
            "global_results": [],
            "current_assessments": [],
            "assessments": []
        }

        self.step = 0

        if load_state:
            self.load(load_path)

        for i, agent in enumerate(self.agents):
            if sub_load_path is not None:
                agent.load_sub(join_path(sub_load_path, "agent-{}".format(i)))

        if store_results:
            local_results = self.records["local_results"]
            global_results = self.records["global_results"]
        else:
            local_results = []
            global_results = []
        assessments = self.records["assessments"]
        current_assessments = self.records["current_assessments"]
        self.latest_policy = [agent.get_initial_policy() for agent in self.agents]
        self.avg_policy = [agent.get_avg_policy() for agent in self.agents]
        self.policy_pool = [[] for _ in self.agents]

        def check_every(every):
            return every is not None and self.step > 0 and self.step % every == 0

        self.push_list = []

        train_info = [[] for _ in range(self.num_agents)]

        last_time = time.time()
        while self.step < max_steps:
            self.step += 1
            assert(len(self.push_list) == 0)
            for i, policy in self.push_list:
                self._push_policy(i, policy)
            for i, agent in enumerate(self.agents):
                for _ in range(train_steps[i]):
                    train_info[i].append(agent.train(i, None, self.step / max_steps, self.step))

            if check_every(test_every):
                # local_result, global_result = self.run_test(test_max_steps)
                # if store_results:
                # local_results.append(local_result)
                # global_results.append(global_result)
                if record_assessment:
                    # rews = self.run_benchmark(1000)
                    # assessment = self.env.assess_strategies([self.statistics.get_avg_strategy(i)
                    #                                          for i in range(self.num_agents)])
                    current_assessment = self.env.assess_strategies([self.latest_policy[i]
                                                                     for i in range(self.num_agents)])
                    print("AVG:")
                    avg_assessment = self.env.assess_strategies([self.avg_policy[i]
                                                                     for i in range(self.num_agents)])
                    # assessment = self.env.assess_strategies([self.latest_policy[i].strategy_fn
                    #                                          for i in range(self.num_agents)])
                    # for i in range(self.num_agents):
                    #     assessment.append(self.env.assess_strategy(i, self.statistics.get_avg_strategy(i)))
                    # exp[0] += 0.633
                    # exp[1] += 2.387
                    assessments.append(avg_assessment)
                    current_assessments.append(current_assessment)
                    # print("Average assessment:", assessment)
                    # print("Current assessment:", current_assessment)
                    # self.run_benchmark()

                now_time = time.time()
                print("\n### Step %d / %d" % (self.step, max_steps), now_time - last_time)
                last_time = now_time

            if check_every(save_every):
                self.save(join_path_and_check(save_path, "step-{}".format(self.step)))
        print("final avg:")
        avg_assessment = self.env.assess_strategies([self.avg_policy[i]
                                                     for i in range(self.num_agents)])
        atk_vn, dfd_vn = self.env.calc_vn(self.avg_policy[0], self.avg_policy[1], 4096, 500)
        for v in atk_vn:
            v.save(save_path)
        dfd_vn.save(save_path)
        # if test_every is not None:
        #     self.statistics.show_statistics()

        # if record_assessment:
        #     final_assessment = self.env.assess_strategies([self.statistics.get_avg_strategy(i, trim_th=1e-3)
        #                                                    for i in range(self.num_agents)], verbose=True)
        #     self.records["final_assessment"] = final_assessment
        #     random_statistics = Statistics(self.env)
        #     random_assessment = self.env.assess_strategies([random_statistics.get_avg_strategy(i)
        #                                                    for i in range(self.num_agents)], verbose=False)
        #     self.records["random_assessment"] = random_assessment

        # self.run_benchmark(10000)
        return self.records, local_results, train_info

    def test(self, *args, **kwargs):
        pass
