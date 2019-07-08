import numpy as np
from env.base_env import BaseEnv
from agent.policy import Policy
import random
import pickle


class Statistics(object):
    def __init__(self, env: BaseEnv, save_file=None):
        self.n_agents = env.num_agents
        self.ob_encoders = env.get_ob_encoders()
        self.ob_namers = env.get_ob_namers()
        self.ac_encoders = env.get_ac_encoders()
        self.n_acs = env.get_n_acs()
        self.ob_maps = None
        self.stats = None
        self.sum_rews = None
        self.tot_steps = None
        self.sum_rews_per_player = None
        if save_file is None:
            self.reset()
        else:
            self.load(save_file)

    def reset(self):
        self.ob_maps = [{} for _ in range(self.n_agents)]
        self.stats = [{} for _ in range(self.n_agents)]
        self.sum_rews = [{} for _ in range(self.n_agents)]
        self.sum_rews_per_player = [0. for _ in range(self.n_agents)]
        self.tot_steps = 0

    def save(self, file):
        raise NotImplementedError
        # if type(file) == str:
        #     file = open(file, "wb")
        # to_export = (self.ob_maps, self.stats, self.sum_rews, self.tot_steps)
        # pickle.dump(to_export, file)

    def load(self, file):
        raise NotImplementedError
        # if type(file) == str:
        #     file = open(file, "rb")
        # self.ob_maps, self.stats, self.sum_rews, self.tot_steps = pickle.load(file)

    def get_update_handler(self):
        def update_handler(last_obs, start, actions, rews, infos, done, obs):
            if start:
                return
            self.tot_steps += 1
            # print(actions, self.ac_encoders)
            eobs = [self.ob_encoders[i](ob) for i, ob in enumerate(last_obs)]
            eacs = [self.ac_encoders[i](ac) for i, ac in enumerate(actions)]
            for i in range(self.n_agents):
                self.sum_rews_per_player[i] += rews[i]
                if eobs[i] not in self.ob_maps[i]:
                    self.ob_maps[i][eobs[i]] = last_obs[i]
                    # print(self.n_acs[i])
                    self.stats[i][eobs[i]] = np.zeros(shape=self.n_acs[i], dtype=np.int32)
                    self.sum_rews[i][eobs[i]] = 0.
                self.stats[i][eobs[i]][eacs[i]] += 1
                self.sum_rews[i][eobs[i]] += rews[i]
        return update_handler

    def get_avg_rew(self, i, ob):
        eob = self.ob_encoders[i](ob)
        if eob not in self.sum_rews[i]:
            return -np.inf
        return self.sum_rews[i][eob] / np.sum(self.stats[i][eob])

    def get_avg_policy(self, i):
        def act_fn(ob):
            eob = self.ob_encoders[i](ob)
            if eob not in self.stats[i]:
                return random.choices(range(self.n_acs[i]))[0]
            else:
                # print(i, self.stats[i][eob])
                return random.choices(range(self.n_acs[i]), weights=self.stats[i][eob])[0]
        return Policy(act_fn)

    def get_avg_strategy(self, i):
        def strategy(ob):
            eob = self.ob_encoders[i](ob)
            if eob not in self.stats[i]:
                return np.ones(self.n_acs[i]) / self.n_acs[i]
            else:
                # print(i, self.stats[i][eob])
                return np.array(self.to_freq(self.stats[i][eob]))
        return strategy

    @staticmethod
    def to_freq(arr):
        return arr / np.sum(arr)

    def show_statistics(self):
        print("Total steps: {}".format(self.tot_steps))
        for i in range(self.n_agents):
            print("\nAgent {}".format(i))
            for eob, ob in self.ob_maps[i].items():
                print(self.ob_namers[i](ob), end='\t')
                print("pi: {0:.2%}".format(np.sum(self.stats[i][eob]) / self.tot_steps), end='\t')
                print("avg_rew: {:+.3f}".format(self.get_avg_rew(i, ob)), end='\t')
                freq = self.to_freq(self.stats[i][eob])
                for j in range(self.n_acs[i]):
                    print("{0:.2%}".format(freq[j]), end=' ')
                print()

    def export_statistics(self):
        result = []
        for i in range(self.n_agents):
            result_map = {}
            for eob, ob in self.ob_maps[i].items():
                result_map[eob] = self.to_freq(self.stats[i][eob])
            result.append(result_map)
        return result

    def get_avg_rews_per_player(self):
        return np.array(self.sum_rews_per_player) / self.tot_steps
