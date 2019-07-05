import random


class Policy(object):
    def __init__(self, act_fn):
        self.act_fn = act_fn

    def act(self, obs):
        action = self.act_fn(obs)
        return action

    def act_clean(self, obs):
        action = self.act_fn(obs)
        if isinstance(action, tuple):
            return action[0]
        else:
            return action


class MixedPolicy(Policy):
    def __init__(self, policies, probabilities):
        self.policies = policies
        self.probabilities = probabilities

        def act_fn(obs):
            p = random.choice(self.policies, self.probabilities)[0]
            return p.act(obs)

        super().__init__(act_fn)

