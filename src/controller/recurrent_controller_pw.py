from env.base_env import BaseEnv
from env.reduced_env import ReducedEnv
from env.monitor_env import MonitorEnv
from controller.base_controller import BaseController
from agent.base_agent import BaseAgent
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
        self.records = None
        self.push_list = None
        self.latest_policy = None
        self.policy_pool = None
        self.avg_policy = None

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
            if version == "latest":
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
        joblib.dump((self.step, self.records), join_path_and_check(save_path, "records.obj"), compress=3)

    def load(self, load_path):
        for i, agent in enumerate(self.agents):
            agent.load(join_path(load_path, "agent-{}".format(i)))
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

        self.records = {
            "local_results": [],
            "global_results": [],
            "current_assessments": [],
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
        current_assessments = self.records["current_assessments"]
        self.latest_policy = [agent.get_initial_policy() for agent in self.agents]
        self.avg_policy = [agent.get_average_policy() for agent in self.agents]
        self.policy_pool = [[] for _ in self.agents]

        def check_every(every):
            return every is not None and self.step > 0 and self.step % every == 0

        self.push_list = []

        train_info = [[] for _ in range(self.num_agents)]

        last_time = time.time()
        while self.step < max_steps:
            self.step += 1
            assert(len(self.push_list) == 0)
            # for i, policy in self.push_list:
            #     self._push_policy(i, policy)
            for i, agent in enumerate(self.agents):
                assert(train_steps[i] == 1)
                for _ in range(train_steps[i]):
                    if sec_prob:
                        env.update_attacker_policy(self.get_policy_with_version(self.policies[0], version="latest"))
                    train_info[i].append(agent.train(i, None, self.step / max_steps, self.step))

            if check_every(test_every):
                # local_result, global_result = self.run_test(test_max_steps)
                # if store_results:
                # local_results.append(local_result)
                # global_results.append(global_result)
                if record_assessment:
                    # rews = self.run_benchmark(1000)
                    # print("Average")
                    # assessment = self.env.assess_strategies([self.statistics.get_avg_strategy(i)
                    #                                          for i in range(self.num_agents)])
                    print("Average network")
                    assessment = self.env.assess_strategies([self.avg_policy[i].strategy_fn
                                                             for i in range(self.num_agents)])
                    print("Current")
                    current_assessment = self.env.assess_strategies([self.latest_policy[i].strategy_fn
                                                             for i in range(self.num_agents)])
                    # assessment = self.env.assess_strategies([self.latest_policy[i].strategy_fn
                    #                                          for i in range(self.num_agents)])
                    # for i in range(self.num_agents):
                    #     assessment.append(self.env.assess_strategy(i, self.statistics.get_avg_strategy(i)))
                    # exp[0] += 0.633
                    # exp[1] += 2.387
                    assessments.append(assessment)
                    current_assessments.append(current_assessment)
                    print("Average assessment:", assessment)
                    print("Current assessment:", current_assessment)
                    # self.run_benchmark()

                now_time = time.time()
                print("\n### Step %d / %d" % (self.step, max_steps), now_time - last_time)
                last_time = now_time

            if check_every(show_every):
                self.show()

            if check_every(save_every):
                self.save(join_path_and_check(save_path, "step-{}".format(self.step)))

        if record_assessment:
            final_assessment = self.env.assess_strategies([self.avg_policy[i].strategy_fn
                                                           for i in range(self.num_agents)], verbose=True)
            self.records["final_assessment"] = final_assessment
            self.records["random_assessment"] = random_assessment

        # self.run_benchmark(10000)
        return self.records, local_results, train_info

    def run_benchmark(self, max_steps=500):
        pass

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
