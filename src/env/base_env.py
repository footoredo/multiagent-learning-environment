import common.tf_util as U
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self, num_agents, observation_spaces, action_spaces):
        assert len(observation_spaces) == num_agents
        assert len(action_spaces) == num_agents

        self.num_agents = num_agents
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def get_observation_space(self, i):
        return self.observation_spaces[i]

    def get_action_space(self, i):
        return self.action_spaces[i]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass


