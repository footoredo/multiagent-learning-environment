from env.base_env import BaseEnv
from gym import spaces


class MatrixEnv(BaseEnv):
    def __init__(self):
        # self.zero_sum = True
        # self.payoff = [[1., -1.], [-1., 1.]]

        # self.zero_sum = False
        # self.payoff = [[[-1., -1.], [-3., 0.]], [[0., -3.], [-2., -2.]]]

        self.zero_sum = True
        self.payoff = [[0., -2.], [-7., 3.]]

        observation_spaces = [spaces.Box(low=-1., high=1., shape=())] * 2
        action_spaces = [spaces.Discrete(2)] * 2

        super().__init__(2, observation_spaces, action_spaces)

    def reset(self):
        return [0., 0.]

    def _get_payoff(self, a):
        if self.zero_sum:
            return [self.payoff[a[0]][a[1]], -self.payoff[a[0]][a[1]]]
        else:
            return [self.payoff[a[0]][a[1]][0], self.payoff[a[0]][a[1]][1]]

    def step(self, actions):
        return ([0., 0.],
                self._get_payoff(actions),
                actions,
                True)
