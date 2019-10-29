from env.base_env import BaseEnv
from .scenario.deceive import Scenario
from multiagent.environment import MultiAgentEnv
from agent.policy import Policy
from gym import spaces
import numpy as np
import subprocess
from copy import deepcopy
from numpy import linalg as LA
import csv
import joblib


def one_hot(n, i):
    x = np.zeros(n)
    x[i] = 1.
    return x


class DeceiveEnv(BaseEnv):
    def __init__(self, n_targets, prior, n_rounds, steps_per_round):
        self.n_targets = n_targets
        self.prior = np.array(prior)
        self.scenario = Scenario(n_targets, 0)
        self.world = self.scenario.make_world()
        self.env = None
        self.n_rounds = n_rounds
        self.steps_per_round = steps_per_round
        self.history = None
        self.record = None
        self.steps_so_far = None
        self.round = None
        self.last_obs_n = None

        atk_init_ob_space = spaces.Box(low=0., high=1., shape=[n_targets * 2])
        dfd_init_ob_space = spaces.Box(low=0., high=1., shape=[n_targets])

        self.ob_length = ob_length = 2 * (2 + n_targets)
        self.ac_length = ac_length = 5

        atk_info_ob_space = spaces.Box(low=0., high=1., shape=[steps_per_round * (ob_length + ac_length) * 2])
        dfd_info_ob_space = spaces.Box(low=0., high=1., shape=[steps_per_round * (ob_length + ac_length) * 2])

        atk_add_ob_space = spaces.Box(low=0., high=1., shape=[ob_length + steps_per_round])
        dfd_add_ob_space = spaces.Box(low=0., high=1., shape=[ob_length + steps_per_round])

        atk_ob_space = (atk_init_ob_space, atk_info_ob_space, atk_add_ob_space)
        dfd_ob_space = (dfd_init_ob_space, dfd_info_ob_space, dfd_add_ob_space)

        ac_space = spaces.Discrete(ac_length)

        super().__init__(num_agents=2,
                         observation_spaces=[atk_ob_space, dfd_ob_space],
                         action_spaces=[ac_space, ac_space])

        self.goal = None

    def _get_atk_init_ob(self):
        return np.concatenate([self.prior, one_hot(self.n_targets, self.goal)])

    def _get_dfd_init_ob(self):
        return self.prior

    def _get_info_ob(self, history):
        # if len(history) == 0:
        #     return np.zeros((0, self.steps_per_round * (self.ob_length + self.ac_length) * 2))
        # else:
        #     return history
        return [np.zeros(self.steps_per_round * (self.ob_length + self.ac_length) * 2)] + history

    def _get_ob(self, obs_n, step, history):
        atk_init_ob = self._get_atk_init_ob()
        dfd_init_ob = self._get_dfd_init_ob()

        atk_add_ob = np.concatenate([obs_n[0], one_hot(self.steps_per_round, step)])
        dfd_add_ob = np.concatenate([obs_n[1], one_hot(self.steps_per_round, step)])

        atk_info_ob = dfd_info_ob = self._get_info_ob(history)

        return [(atk_init_ob, atk_info_ob, atk_add_ob), (dfd_init_ob, dfd_info_ob, dfd_add_ob)]

    def reset(self, verbose=False):
        world = self.world
        scenario = self.scenario
        self.goal = scenario.goal = np.random.choice(self.n_targets, p=self.prior)
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                                 shared_viewer=False)
        self.env.discrete_action_input = True

        self.history = []
        self.record = np.zeros(self.steps_per_round * (self.ob_length + self.ac_length) * 2)
        self.steps_so_far = 0
        self.round = 0
        obs_n = self.env.reset()
        self.last_obs_n = obs_n

        return self._get_ob(obs_n, self.steps_so_far, self.history), None, None

    def step(self, actions, action_probs):
        if self.steps_so_far < self.steps_per_round:
            s = self.steps_so_far * (self.ob_length + self.ac_length) * 2
            # print(self.record.shape, s)
            self.record[s: s + self.ob_length] = self.last_obs_n[0]
            s += self.ob_length
            self.record[s + actions[0]] = 1.
            s += self.ac_length
            self.record[s: s + self.ob_length] = self.last_obs_n[1]
            s += self.ob_length
            self.record[s + actions[1]] = 1.

        self.steps_so_far += 1
        obs_n, reward_n, done_n, info_n = self.env.step(actions)

        if self.steps_so_far == self.steps_per_round:
            obs_n = self.env.reset()
            self.steps_so_far = 0
            self.round += 1
            self.history.append(np.copy(self.record))
            self.record = np.zeros(self.steps_per_round * (self.ob_length + self.ac_length) * 2)

        self.last_obs_n = obs_n

        return self._get_ob(obs_n, self.steps_so_far, self.history), reward_n, None, self.round >= self.n_rounds, None, None

    def simulate(self, strategies, verbose=False):
        atk_strategy, dfd_strategy = strategies
        ob, _, _ = self.reset()
        atk_rew = 0.
        dfd_rew = 0.
        if verbose:
            print("\nSimulation starts.")
            # self.env.render()
        while True:
            atk_s = atk_strategy(ob[0])
            dfd_s = dfd_strategy(ob[1])
            atk_a = np.random.choice(5, p=atk_s)
            dfd_a = np.random.choice(5, p=dfd_s)
            if verbose:
                print("Attacker: {} -> {} -> {}".format(ob[0], atk_s, atk_a))
                print("Defender: {} -> {} -> {}".format(ob[1], dfd_s, dfd_a))
            ob, rew, _, done, _, _ = self.step([atk_a, dfd_a], None)
            # self.env.render()

            atk_rew += rew[0]
            dfd_rew += rew[1]
            if done:
                break
        if verbose:
            print("Simulation ends.", [atk_rew, dfd_rew])

        return [atk_rew, dfd_rew]

    def assess_strategies(self, strategies, debug=False):
        trials = 1
        atk_rew = 0.
        dfd_rew = 0.
        for _ in range(trials):
            a, d = self.simulate(strategies, verbose=True)
            atk_rew += a
            dfd_rew += d

        return [atk_rew / trials, dfd_rew / trials]
