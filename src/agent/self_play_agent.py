from agent.base_agent import BaseAgent
from agent.policy import Policy
import numpy as np
import math
from copy import deepcopy


class SelfPlayAgent(BaseAgent):
    def __init__(self, n_slots, handlers):
        self.n_slots = n_slots
        self.history = np.zeros(shape=2, dtype=np.int32)
        self.push, self.pull = handlers

    def get_policy(self):
        history = deepcopy(self.history)

        def act(obs):
            if np.sum(history) == 0:
                return np.random.choice(2)
            else:
                return np.random.choice(2, p=history / np.sum(history))

        return Policy(act_fn=act)

    def get_initial_policy(self):
        return self.get_policy()

    def get_final_policy(self):
        return self.get_policy()

    def train(self):
        env = self.pull("latest")

        value0 = 0.
        for _ in range(20):
            env.reset()
            action = 0
            _, rew, _, _ = env.step(action)
            value0 += rew

        value1 = 0.
        for _ in range(20):
            env.reset()
            action = 1
            _, rew, _, _ = env.step(action)
            value1 += rew

        br = np.random.choice(2) if math.isclose(value0, value1) else int(value1 > value0)
        self.history[br] += 1

        self.push(self.get_policy())

    def get_config(self):
        pass
