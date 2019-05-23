from agent.base_agent import BaseAgent
from agent.policy import Policy
import numpy as np
from copy import deepcopy


class InfantAgent(BaseAgent):
    def __init__(self, n_slots, handlers):
        self.n_slots = n_slots
        self.values = np.random.rand(2) * 2. - 1.
        self.push, self.pull = handlers

    def softmax(self, values):
        return np.exp(values) / np.sum(np.exp(values))

    def get_policy(self):
        values = deepcopy(self.values)

        def act(obs):
            return np.random.choice(2, p=self.softmax(values))

        return Policy(act_fn=act)

    def get_initial_policy(self):
        return self.get_policy()

    def get_final_policy(self):
        return self.get_policy()

    def train(self):
        env = self.pull()
        env.reset()
        policy = self.get_policy()
        for _ in range(100):
            action = policy.act(None)
            _, rew, _, _ = env.step(action)
            self.values[action] = self.values[action] * .9 + rew * .1
            env.reset()
        self.push(self.get_policy())

    def get_config(self):
        pass
