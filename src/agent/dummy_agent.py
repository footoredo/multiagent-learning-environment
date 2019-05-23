from agent.base_agent import BaseAgent
from agent.policy import Policy
import random


class DummyAgent(BaseAgent):
    def __init__(self, n_slots):
        self.n_slots = n_slots

    def get_initial_policy(self):
        def random_policy(_):
            return random.randrange(0, self.n_slots), None

        def fixed_policy(_):
            return 0

        return Policy(fixed_policy)

    def get_final_policy(self):
        return self.get_initial_policy()

    def train(self):
        pass

    def get_config(self):
        return {}
