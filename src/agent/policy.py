class Policy:
    def __init__(self, act_fn):
        self.act_fn = act_fn

    def act(self, obs):
        action = self.act_fn(obs)

    def act_clean(self, obs):
        action = self.act_fn(obs)
        if isinstance(action, tuple):
            return action[0]
        else:
            return action

