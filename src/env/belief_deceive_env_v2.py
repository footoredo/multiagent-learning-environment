from env.base_env import BaseEnv
from .scenario.deceive_grid import Scenario
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
    if i < n:
        x[i] = 1.
    return x


class DeceiveEnv(BaseEnv):
    def __init__(self, n_targets, prior, n_steps, steps_per_round, size=3, random_prior=True, is_atk=False):
        self.n_targets = n_targets
        self.prior = np.array(prior)
        self.random_prior = random_prior
        self.size = size
        self.scenario = Scenario(n_targets, 0, size=size, random=True)
        self.world = self.scenario.make_world()
        self.env = None
        self.n_steps = n_steps
        self.steps_per_round = steps_per_round
        self.history = None
        self.record = None
        self.steps_so_far = None
        self.round = None
        self.last_obs_n = None
        self.is_atk = is_atk

        self.n_rounds = (self.n_steps + self.steps_per_round - 1) // self.steps_per_round

        self.ob_length = ob_length = self.scenario.ob_length
        self.ac_length = ac_length = 5

        atk_ob_space = spaces.Box(low=0., high=1., shape=[n_targets + ob_length])
        dfd_ob_space = spaces.Box(low=0., high=1., shape=[ob_length])

        ac_space = spaces.Discrete(ac_length)

        super().__init__(num_agents=2,
                         observation_spaces=[atk_ob_space, dfd_ob_space],
                         action_spaces=[ac_space, ac_space])

        self.goal = None
        self.belief = None
        self.rounds_so_far = None

        self.atk_policy = None
        self.dfd_policy = None

    def get_atk_env(self):
        return DeceiveEnv(self.n_targets, self.prior, self.n_steps, self.steps_per_round,
                          self.size, self.random_prior, True)

    def update_policy(self, i, policy):
        if i == 0:
            self.atk_policy = policy
        else:
            self.dfd_policy = policy

    def _get_atk_ob(self, t, belief, n, world_obs, s):
        return np.concatenate([belief, one_hot(self.n_steps, s), one_hot(self.n_targets, t), world_obs])

    def _get_dfd_ob(self, belief, n, world_obs, s):
        return np.concatenate([belief, one_hot(self.n_steps, s), world_obs])

    def _get_ob(self, t, belief, n, world_obs_n, s):
        return [self._get_atk_ob(t, belief, n, world_obs_n[0], s), self._get_dfd_ob(belief, n, world_obs_n[1], s)]

    def get_type_space(self, i):
        return spaces.Discrete(self.n_targets)

    def generate_prior(self):
        x = [0.] + sorted(np.random.rand(self.n_targets - 1).tolist()) + [1.]
        prior = np.zeros(self.n_targets)
        for i in range(self.n_targets):
            prior[i] = x[i + 1] - x[i]
        return prior

    def reset(self, benchmark=False, verbose=False):
        scenario = self.scenario
        if self.random_prior:
            self.prior = self.generate_prior()
        if self.is_atk:
            self.goal = scenario.goal = np.random.choice(self.n_targets)
        else:
            self.goal = scenario.goal = np.random.choice(self.n_targets, p=self.prior)
        world = self.world
        if benchmark:
            # world.agents[0].state.p_pos = np.array([-0.5, -0.5]) # for step-1
            # world.agents[1].state.p_pos = np.array([-0.0, 0.0])
            # world.agents[0].state.p_pos = np.array([-0.5, 0.0]) # for step-2
            # world.agents[1].state.p_pos = np.array([0.5, 0.0])
            # world.agents[0].state.p_pos = np.array([-0.0, 0.0]) # for step-1
            # world.agents[1].state.p_pos = np.array([-0.0, 0.0])
            world.agents[0].state.p_pos = np.array([1.0, 0.0])
            world.agents[1].state.p_pos = np.array([0.0, 0.0])
            # world.agents[0].state.p_pos = np.array([1.0, 1.0])
            # world.agents[1].state.p_pos = np.array([1.0, 0.0])
            self.goal = scenario.goal = 1
            scenario.change_position = False
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None)
        self.env.discrete_action_input = True

        self.history = []
        self.steps_so_far = 0
        self.round = 0
        obs_n = self.env.reset()
        self.belief = np.copy(self.prior)
        self.last_obs_n = obs_n

        return self._get_ob(self.goal, self.belief, self.round, obs_n, self.steps_so_far), None, None

    def update_belief(self, belief, probs):
        tmp = belief * probs
        if np.sum(tmp) < 1e-2:
            return np.ones(self.n_targets) / self.n_targets
        return tmp / np.sum(tmp)

    def step(self, actions, action_probs, frames=None):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        if frames is not None:
            frames.append(self.env.render('rgb_array')[0])
        # for i in range(2):
        #     reward_n[i] *= self.round + 1
        atk_touching = [self.scenario.is_touching(self.world.agents[0], self.world.landmarks[i]) for i in range(self.n_targets)]
        dfd_touching = [self.scenario.is_touching(self.world.agents[1], self.world.landmarks[i]) for i in range(self.n_targets)]

        probs = [self.atk_policy.prob(self._get_atk_ob(t, self.belief, self.round, self.last_obs_n[0],
                                                       self.steps_so_far), actions[0]) for t in range(self.n_targets)]
        self.belief = self.update_belief(self.belief, np .array(probs))
        # print(self.belief)

        self.steps_so_far += 1
        sub_done = False
        for i in range(2):
            for _ in range(self.round):
                reward_n[i] *= 2
        if self.steps_so_far % self.steps_per_round == 0:
            # for i, landmark in enumerate(self.world.landmarks):
            #     if self.scenario.is_touching(self.world.agents[0], landmark):
            #         if self.scenario.is_touching(self.world.agents[1], landmark):
            #             reward_n[1] += 20
            #             reward_n[0] -= 5
            #         elif i == self.goal:
            #             reward_n[0] += 20
            #         else:
            #             reward_n[0] += 5

            # if self.scenario.is_touching(self.world.agents[0], self.world.landmarks[self.goal]):
            #     reward_n[0] += 20
            #
            # if self.scenario.is_touching(self.world.agents[1], self.world.landmarks[self.goal]):
            #     reward_n[1] += 20

            obs_n = self.env.reset()
            if frames is not None and self.steps_so_far < self.n_steps:
                frames.append(self.env.render('rgb_array')[0])
            # self.steps_so_far = 0
            self.round += 1
            # self.history.append(np.copy(self.record))
            # self.record = np.zeros(self.steps_per_round * (self.ob_length + self.ac_length) * 2)
            sub_done = True

        self.last_obs_n = obs_n

        return self._get_ob(self.goal, self.belief, self.round, obs_n, self.steps_so_far), reward_n, \
               [self.goal, self.goal], self.steps_so_far >= self.n_steps, [sub_done, sub_done], [atk_touching, dfd_touching]

    def simulate(self, strategies, verbose=False, save_dir=None, prior=None, benchmark=False):
        rp = self.random_prior
        if prior is not None:
            self.prior = np.copy(prior)
            self.random_prior = False
        atk_policy, dfd_policy = strategies
        atk_strategy, dfd_strategy = atk_policy.strategy_fn, dfd_policy.strategy_fn
        ob, _, _ = self.reset(benchmark)
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
                print("Attacker: {} -> {} -> {}".format(ob[0], atk_s, atk_a))
                print("Defender: {} -> {} -> {}".format(ob[1], dfd_s, dfd_a))
                # if sub_steps == 0:
                #     print("tpred:", atk_policy.tpred(ob[0]))
                #     print("tpred:", dfd_policy.tpred(ob[1]))
            ob, rew, _, done, sub_done, touching = self.step([atk_a, dfd_a], None, frames=frames if verbose else None)
            sub_steps += 1
            at, dt = touching

            for i in range(self.n_targets):
                atk_touching[steps][i] = atk_touching[steps][i] or at[i]
                dfd_touching[steps][i] = dfd_touching[steps][i] or dt[i]

            if sub_done[0]:
                steps += 1
                sub_steps = 0
            # if verbose:
            #     frames.append(self.env.render('rgb_array')[0])
            # self.env.render()

            atk_rew += rew[0]
            dfd_rew += rew[1]
            if done:
                break

        atk_same = True
        dfd_same = True
        atk_choice_0 = np.argmax(atk_touching[0])
        for i in range(1, self.n_rounds):
            atk_same = atk_same and np.argmax(atk_touching[i]) == atk_choice_0
            dfd_same = dfd_same and np.argmax(dfd_touching[i]) == atk_choice_0

        if verbose:
            print("Simulation ends.", [atk_rew, dfd_rew])
            imageio.mimsave(save_dir, frames, duration=1 / 3)
            for viewer in self.env.viewers:
                viewer.close()

        atk_touching = [[atk_touching[i][self.goal], sum(atk_touching[i])] for i in range(self.n_rounds)]
        dfd_touching = [[dfd_touching[i][self.goal], sum(dfd_touching[i])] for i in range(self.n_rounds)]
        self.random_prior = rp
        return atk_rew, dfd_rew, atk_touching, dfd_touching, atk_same, dfd_same

    def _assess_strategies(self, strategies, trials=100, debug=False, prior=None):
        atk_rew = 0.
        dfd_rew = 0.
        atk_right = [[0, 0] for _ in range(self.n_rounds)]
        dfd_right = [[0, 0] for _ in range(self.n_rounds)]
        atk_same = 0
        dfd_same = 0
        for _ in range(trials):
            a, d, at, dt, _as, ds = self.simulate(strategies, verbose=False, prior=prior)
            atk_rew += a
            dfd_rew += d
            atk_same += _as
            dfd_same += ds
            for i in range(self.n_rounds):
                for j in range(2):
                    atk_right[i][j] += at[i][j]
                    dfd_right[i][j] += dt[i][j]

        print("Attacker rights:", np.array(atk_right) / trials, [atk_right[i][0] / atk_right[i][1] for i in range(self.n_rounds)])
        print("Defender rights:", np.array(dfd_right) / trials, [dfd_right[i][0] / dfd_right[i][1] for i in range(self.n_rounds)])
        print("Attacker same:", atk_same / trials)
        print("Defender same:", dfd_same / trials)
        print("Attacker reward:", atk_rew / trials)
        print("Defender reward:", dfd_rew / trials)

        return [atk_rew / trials, dfd_rew / trials]

    def assess_strategies(self, strategies, trials=100, debug=False):
        if self.random_prior:
            for p in range(11):
                prior = np.array([p / 10, 1 - p / 10])
                print("Prior:", prior)
                self._assess_strategies(strategies, trials, debug, prior)
                print("")
        else:
            return self._assess_strategies(strategies, trials, debug)
