from env.base_env import BaseEnv, BaseEnvWrapper


class ReducedEnv(BaseEnvWrapper):
    def __init__(self, base_env: BaseEnv, fixed_indices, fixed_policies, allow_single=True):
        self.fixed_indices = set(fixed_indices)
        self.fixed_policies = fixed_policies
        self.unfixed_indices = set(range(base_env.num_agents)) - self.fixed_indices
        self.is_single = len(self.unfixed_indices) == 1 and allow_single
        self.single_index = list(self.unfixed_indices)[0] if self.is_single else None
        super().__init__(base_env,
                         num_agents=len(self.unfixed_indices),
                         observation_spaces=self.choose_unfixed(base_env.observation_spaces),
                         action_spaces=self.choose_unfixed(base_env.action_spaces))
        self.fixed_obs = []

    @property
    def observation_space(self):
        if not self.is_single:
            raise Exception("not single")
        return self.observation_spaces[0]

    @property
    def action_space(self):
        if not self.is_single:
            raise Exception("not single")
        return self.action_spaces[0]

    def choose_unfixed(self, arr):
        # if isinstance(arr, tuple):
        #     return tuple([self.choose_unfixed(_arr) for _arr in arr])
        return [arr[i] for i in self.unfixed_indices]

    def choose_fixed(self, arr):
        # if isinstance(arr, tuple):
        #     return tuple([self.choose_fixed(_arr) for _arr in arr])
        return [arr[i] for i in self.fixed_indices]

    def merge(self, arr_fixed, arr_unfixed):
        # print(len(arr_fixed), len(arr_unfixed))
        arr = [None for _ in range(self.base_env.num_agents)]
        for i, index in enumerate(self.fixed_indices):
            arr[index] = arr_fixed[i]
        for i, index in enumerate(self.unfixed_indices):
            arr[index] = arr_unfixed[i]
        return arr

    def choose_unfixed_single(self, arr):
        # if isinstance(arr, tuple):
        #     return tuple([self.choose_unfixed_single(_arr) for _arr in arr])
        arr = self.choose_unfixed(arr)
        if self.is_single:
            return arr[0]

    def merge_single(self, arr_fixed, arr_unfixed):
        if self.is_single:
            return self.merge(arr_fixed, [arr_unfixed])
        else:
            return self.merge(arr_fixed, arr_unfixed)

    def reset(self, debug=False):
        # print("ASd")
        obs = self.base_env.reset(debug)
        self.fixed_obs = self.choose_fixed(obs)
        return self.choose_unfixed_single(obs)

    def step(self, actions):
        fixed_actions = [self.fixed_policies[i].act_clean(self.fixed_obs[i]) for i in range(len(self.fixed_policies))]
        ret = super().step(self.merge_single(fixed_actions, actions))
        # print(ret)
        obs, rews, infos, done = ret
        self.fixed_obs = self.choose_fixed(obs)
        return self.choose_unfixed_single(obs), self.choose_unfixed_single(rews), self.choose_unfixed_single(infos), done
