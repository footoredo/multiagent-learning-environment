from env.base_env import BaseEnvWrapper


class MonitorEnv(BaseEnvWrapper):
    def __init__(self, base_env, update_handler):
        super().__init__(base_env)
        # print(update_handler)
        self.update_handler = update_handler

    def reset(self):
        obs = self.base_env.reset()
        # print("ASd")
        self.update_handler(
            start=True,
            actions=None,
            rews=None,
            infos=None,
            done=False,
            obs=obs
        )
        return obs

    def step(self, actions):
        obs, rews, infos, done = self.base_env.step(actions)
        self.update_handler(
            start=False,
            actions=actions,
            rews=rews,
            infos=infos,
            done=done,
            obs=obs
        )
        return obs, rews, infos, done
