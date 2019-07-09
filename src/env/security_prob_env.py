from env.base_env import BaseEnvWrapper
from env.security_env import SecurityEnv
from agent.policy import Policy
from gym import spaces
import numpy as np


class SecurityProbEnv(BaseEnvWrapper):
    def __init__(self, sec_env: SecurityEnv):
        # atk_ob_space = spaces.Box(low=0., high=1., shape=[sec_env.n_types * 2 + sec_env.n_rounds])
        # dfd_ob_space = spaces.Box(low=0., high=1., shape=[sec_env.n_types + sec_env.n_rounds])
        atk_ob_space = spaces.Box(low=0., high=1., shape=[sec_env.n_types + 1])
        dfd_ob_space = spaces.Box(low=0., high=1., shape=[1])
        super().__init__(sec_env, observation_spaces=[atk_ob_space, dfd_ob_space])
        self.sec_env = sec_env

        self.prob = None
        self.rounds_so_far = None
        self.attacker_type = None
        self.attacker_policy = None

    @staticmethod
    def _to_one_hot(i, n):
        ret = np.zeros(shape=n, dtype=np.float32)
        if i < n:
            ret[i] = 1.0
        return ret

    def _gen_dfd_ob(self):
        # print(self.rounds_so_far, self.sec_env.n_rounds)
        # return np.concatenate([self.prob, SecurityProbEnv._to_one_hot(self.rounds_so_far, self.sec_env.n_rounds)])
        return np.zeros(1)

    def _gen_atk_ob(self, atk_type=None):
        if atk_type is None:
            atk_type = self.attacker_type
        return np.concatenate([SecurityProbEnv._to_one_hot(atk_type, self.sec_env.n_types),
                               self._gen_dfd_ob()])

    def _gen_ob(self, atk_type=None):
        return [self._gen_atk_ob(atk_type=atk_type), self._gen_dfd_ob()]

    def update_attacker_policy(self, attacker_policy: Policy):
        # print("UPDATED!")
        self.attacker_policy = attacker_policy

    def reset(self, debug=False):
        self.sec_env.reset()
        self.prob = np.array(self.sec_env.prior)
        self.rounds_so_far = 0
        self.attacker_type = self.sec_env.atk_type

        return self._gen_ob()

    def step(self, actions):
        _, rews, infos, done = self.sec_env.step(actions)
        likelihood = np.zeros(shape=self.sec_env.n_types, dtype=np.float32)
        for t in range(self.sec_env.n_types):
            likelihood[t] = self.attacker_policy.prob(self._gen_atk_ob(atk_type=t), actions[0])
            # print(t, likelihood[t])
        self.prob *= likelihood
        self.prob /= np.sum(self.prob)
        # print(self.prob)
        self.rounds_so_far += 1
        # print(self.rounds_so_far, done)
        return self._gen_ob(), rews, infos, done

    def calc_exploitability(self, i, strategy):
        raise NotImplementedError

