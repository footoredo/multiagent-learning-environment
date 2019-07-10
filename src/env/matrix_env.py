from env.base_env import BaseEnv
from gym import spaces
import numpy as np


class MatrixEnv(BaseEnv):
    def __init__(self):
        # self.zero_sum = True
        # self.payoff = [[1., -1.], [-1., 1.]]

        # self.zero_sum = False
        # self.payoff = [[[-1., -1.], [-3., 0.]], [[0., -3.], [-2., -2.]]]

        self.zero_sum = True
        # self.payoff = np.array([[1., 3., 2], [0., 5., 1], [3, 0, 2]])
        self.payoff = np.array([[2, 3, 4], [5, -1, -7]])

        self.na = list(self.payoff.shape)

        observation_spaces = [spaces.Box(low=-1., high=1., shape=[1])] * 2
        action_spaces = [spaces.Discrete(self.na[0]), spaces.Discrete(self.na[1])]

        super().__init__(2, observation_spaces, action_spaces)

    def reset(self, debug=False):
        return [np.zeros(1), np.zeros(1)]

    def _get_payoff(self, a):
        if self.zero_sum:
            return [self.payoff[a[0]][a[1]], -self.payoff[a[0]][a[1]]]
        else:
            return [self.payoff[a[0]][a[1]][0], self.payoff[a[0]][a[1]][1]]

    def step(self, actions):
        return (self.reset(),
                self._get_payoff(actions),
                actions,
                True)

    def calc_exploitability(self, i, strategy):
        prob = strategy(np.zeros(1))
        j = 1 if i == 0 else 0
        exp = -1e100
        for aj in range(self.na[j]):
            ret = 0.
            for ai in range(self.na[i]):
                a = [None, None]
                a[i] = ai
                a[j] = aj
                tmp = self._get_payoff(a)[j]
                ret += prob[ai] * tmp
            exp = max(exp, ret)
        return exp
