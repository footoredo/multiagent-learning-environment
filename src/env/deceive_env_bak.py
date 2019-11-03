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
import os
import imageio


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

    def get_type_space(self, i):
        return spaces.Discrete(self.n_targets)

    def reset(self, verbose=False):
        scenario = self.scenario
        self.goal = scenario.goal = np.random.choice(self.n_targets, p=self.prior)
        world = self.world
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None)
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
        # for i in range(2):
        #     reward_n[i] *= self.round + 1
        atk_touching = [self.scenario.is_touching(self.world.agents[0], self.world.landmarks[i]) for i in range(self.n_targets)]
        dfd_touching = [self.scenario.is_touching(self.world.agents[1], self.world.landmarks[i]) for i in range(self.n_targets)]

        sub_done = False
        if self.steps_so_far == self.steps_per_round:
            obs_n = self.env.reset()
            self.steps_so_far = 0
            self.round += 1
            self.history.append(np.copy(self.record))
            self.record = np.zeros(self.steps_per_round * (self.ob_length + self.ac_length) * 2)
            sub_done = True

        self.last_obs_n = obs_n

        return self._get_ob(obs_n, self.steps_so_far, self.history), reward_n, [self.goal, self.goal], \
               self.round >= self.n_rounds, [sub_done, sub_done], [atk_touching, dfd_touching]

    def simulate(self, strategies, verbose=False, save_dir=None):
        atk_strategy, dfd_strategy = strategies
        ob, _, _ = self.reset()
        atk_rew = 0.
        dfd_rew = 0.
        atk_touching = [[0 for _ in range(self.n_targets)] for _ in range(self.n_rounds)]
        dfd_touching = [[0 for _ in range(self.n_targets)] for _ in range(self.n_rounds)]
        if verbose:
            print("\nSimulation starts.")
            # self.env.render()
        frames = list()
        if verbose:
            frames.append(self.env.render('rgb_array')[0])
        steps = 0
        sub_steps = 0
        while True:
            atk_s = atk_strategy(ob[0])
            dfd_s = dfd_strategy(ob[1])
            atk_a = np.random.choice(5, p=atk_s)
            dfd_a = np.random.choice(5, p=dfd_s)
            if verbose:
                print("Round {} Step {}".format(steps, sub_steps))
                print("Attacker: {} -> {}".format(atk_s, atk_a))
                print("Defender: {} -> {}".format(dfd_s, dfd_a))
            ob, rew, _, done, sub_done, touching = self.step([atk_a, dfd_a], None)
            sub_steps += 1
            at, dt = touching

            for i in range(self.n_targets):
                atk_touching[steps][i] += at[i]
                dfd_touching[steps][i] += dt[i]

            if sub_done[0]:
                steps += 1
                sub_steps = 0
            if not done and verbose:
                frames.append(self.env.render('rgb_array')[0])
            # self.env.render()

            atk_rew += rew[0]
            dfd_rew += rew[1]
            if done:
                break
        if verbose:
            print("Simulation ends.", [atk_rew, dfd_rew])
            imageio.mimsave(save_dir, frames, duration=1 / 5)
            for viewer in self.env.viewers:
                viewer.close()

        atk_touching = [[atk_touching[i][self.goal], sum(atk_touching[i])] for i in range(self.n_rounds)]
        dfd_touching = [[dfd_touching[i][self.goal], sum(dfd_touching[i])] for i in range(self.n_rounds)]
        return atk_rew, dfd_rew, atk_touching, dfd_touching

    def assess_strategies(self, strategies, trials=100, debug=False):
        atk_rew = 0.
        dfd_rew = 0.
        atk_right = [[0, 0] for _ in range(self.n_rounds)]
        dfd_right = [[0, 0] for _ in range(self.n_rounds)]
        for _ in range(trials):
            a, d, at, dt = self.simulate(strategies, verbose=False)
            atk_rew += a
            dfd_rew += d
            for i in range(self.n_rounds):
                for j in range(2):
                    atk_right[i][j] += at[i][j]
                    dfd_right[i][j] += dt[i][j]

        print("Attacker rights:", np.array(atk_right) / trials)
        print("Defender rights:", np.array(dfd_right) / trials)
        print("Attacker reward:", atk_rew / trials)
        print("Defender reward:", dfd_rew / trials)

        return [atk_rew / trials, dfd_rew / trials]
