from env.base_env import BaseEnvWrapper


class MonitorEnv(BaseEnvWrapper):
    def __init__(self, base_env, update_handlers):
        super().__init__(base_env)
        # print(update_handler)
        self.update_handlers = update_handlers
        self.last_obs = None

    def update(self, *args, **kwargs):
        if type(self.update_handlers) == list:
            for update_handler in self.update_handlers:
                update_handler(*args, **kwargs)
        elif self.update_handlers is not None:
            self.update_handlers(*args, **kwargs)

    def reset(self, debug=False):
        obs, probs = self.base_env.reset(debug)
        # print("ASd")
        self.update(
            last_obs=None,
            start=True,
            actions=None,
            rews=None,
            infos=None,
            done=False,
            obs=obs
        )
        self.last_obs = obs
        return obs, probs

    def step(self, actions, action_probs):
        obs, rews, infos, done, probs = self.base_env.step(actions, action_probs)
        self.update(
            last_obs=self.last_obs,
            start=False,
            actions=actions,
            rews=rews,
            infos=infos,
            done=done,
            obs=obs
        )
        self.last_obs = obs
        return obs, rews, infos, done, probs
