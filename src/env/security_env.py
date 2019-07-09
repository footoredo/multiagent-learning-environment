from env.base_env import BaseEnv
from agent.policy import Policy
from gym import spaces
import numpy as np
import subprocess
from copy import deepcopy
import csv
import pickle


class GambitSolver:
    def __init__(self, n_types, n_slots, n_stages, prior, payoff):
        self.n_types = n_types
        self.n_slots = n_slots
        self.n_stages = n_stages
        self.prior = prior
        self.payoff = payoff
        self.infosets = None
        self.file = None
        self.solution = None

    def generate(self):
        self.infosets = [[], []]
        self.file = open("game.efg", "w")

        self.println("EFG 2 R \"Bayesian security game, %d stage(s)\" { \"Attacker\" \"Defender\" }" % self.n_stages)
        # self.println("c \"Chance\" 1 \"\" { \"type 0\" %.5f \"type 1\" %.5f } 0" % (self.prior[0], self.prior[1]))
        self.print("c \"Chance\" 1 \"\" { ")
        for t in range(self.n_types):
            self.print("\"type %d\" %.5f " % (t, self.prior[t]))
        self.println("} 0")

        for t in range(self.n_types):
            self.recursive(self.n_stages, t, -1, -1, [t], [])

        self.file.close()

    def solve(self):
        # command = subprocess.Popen(['ls', '-a'], stdout=subprocess.PIPE)
        command = subprocess.Popen(['gambit-lcp', '-d', '5', '-P', '-q', 'game.efg'],
                                   stdout=subprocess.PIPE)
        out = command.stdout.readlines()[0]
        self.solution = list(map(float, out.split(b',')[1:]))

    def get_profile(self, player, history):
        position = self.get_infoset(player, history) - 1
        if player == 1:
            position += len(self.infosets[0])

        return self.solution[position * self.n_slots:][:self.n_slots]

    def print(self, s):
        self.file.write(s)

    def println(self, s):
        self.print(s + '\n')

    @staticmethod
    def infoset_to_str(h, with_type):
        if len(h) == 0:
            return ""
        if with_type:
            return str(h[0]) + ":" + GambitSolver.infoset_to_str(h[1:], False)
        else:
            # print(h)
            return "(" + str(h[0]) + "," + str(h[1]) + ")" + ":" + GambitSolver.infoset_to_str(h[2:], False)
            # return "(" + str(h[0]) + ")" + ":" + GambitSolver.infoset_to_str(h[1:], False)

    def get_infoset(self, player, history):
        if history not in self.infosets[player]:
            self.infosets[player].append(history)
        return self.infosets[player].index(history) + 1

    def print_outcome(self, t, i, j):
        on = t * self.n_slots * self.n_slots + i * self.n_slots + j + 1
        ar = self.payoff[t, i, j, 0]
        dr = self.payoff[t, i, j, 1]
        self.println(" %d \"(%d,%d,%d)\" { %.5f, %.5f }" % (on, t, i, j, ar, dr))

    def recursive(self, remain_stage, t, li, lj, h0, h1):
        assert (remain_stage >= 0)

        if remain_stage == 0:
            self.print("t \"\"")
            self.print_outcome(t, li, lj)
            return

        self.print("p \"\" 1 %d \"%s\"" % (self.get_infoset(0, h0), GambitSolver.infoset_to_str(h0, True)))
        self.print(" {")
        for i in range(self.n_slots):
            self.print(" \"s%d\"" % i)
        self.print(" }")

        if li >= 0 and lj >= 0:
            self.print_outcome(t, li, lj)
        else:
            self.println(" 0")

        for i in range(self.n_slots):
            self.print("p \"\" 2 %d \"%s\"" % (self.get_infoset(1, h1), GambitSolver.infoset_to_str(h1, False)))
            self.print(" {")
            for j in range(self.n_slots):
                self.print(" \"s%d\"" % j)
            self.print(" }")
            self.println(" 0")

            for j in range(self.n_slots):
                self.recursive(remain_stage - 1, t, i, j, h0 + [i, j], h1 + [i, j])


class SecurityEnv(BaseEnv):
    def __init__(self, n_slots, n_types, prior, n_rounds, value_range=10., zero_sum=False, seed=None):
        self.n_slots = n_slots
        self.n_types = n_types
        self.prior = prior if prior is not None else np.random.rand(n_types)
        self.prior /= np.sum(self.prior)
        self.n_rounds = n_rounds
        self.zero_sum = zero_sum
        self.seed = seed

        self.ob_shape = (n_rounds - 1, 2, n_slots + 1)
        self.ob_len = np.prod(self.ob_shape)
        atk_ob_space = spaces.Box(low=0., high=1., shape=[n_types + self.ob_len])
        dfd_ob_space = spaces.Box(low=0., high=1., shape=[1 + self.ob_len])
        # print(dfd_ob_space)
        ac_space = spaces.Discrete(n_slots)
        super().__init__(num_agents=2,
                         observation_spaces=[atk_ob_space, dfd_ob_space],
                         action_spaces=[ac_space, ac_space])

        self.rounds_so_far = None
        self.ac_history = None
        self.atk_type = None
        self.type_ob = None

        if seed == "benchmark":
            assert n_slots == 2 and n_rounds == 1 and n_types == 2
            self.atk_rew = np.array([[2., 1.], [1., 2.]])
            self.atk_pen = np.array([[-1., -1.], [-1., -1.]])
            self.dfd_rew = np.array([1., 1.])
            self.dfd_pen = np.array([-1., -1.])
        else:
            if seed is not None:
                np.random.seed(seed)
            self.atk_rew = np.random.rand(n_types, n_slots) * value_range
            self.atk_pen = -np.random.rand(n_types, n_slots) * value_range
            self.dfd_rew = np.random.rand(n_slots) * value_range
            self.dfd_pen = -np.random.rand(n_slots) * value_range

        self.payoff = np.zeros((n_types, n_slots, n_slots, 2), dtype=np.float32)
        for t in range(n_types):
            for i in range(n_slots):
                for j in range(n_slots):
                    if i == j:
                        self.payoff[t, i, j, 0] = self.atk_pen[t, i]
                        if zero_sum:
                            self.payoff[t, i, j, 1] = -self.atk_pen[t, i]
                        else:
                            self.payoff[t, i, j, 1] = self.dfd_rew[j]
                    else:
                        self.payoff[t, i, j, 0] = self.atk_rew[t, i]
                        if zero_sum:
                            self.payoff[t, i, j, 1] = -self.atk_rew[t, i]
                        else:
                            self.payoff[t, i, j, 1] = self.dfd_pen[j]

        # print(self.payoff[0, :, :, 0])

        self.gambit_solver = GambitSolver(n_slots=n_slots, n_types=n_types, n_stages=n_rounds, payoff=self.payoff, prior=self.prior)
        self.gambit_solver.generate()

        self.attacker_exploitability_calculator = self._AttackerExploitabilityCalculator(self)
        self.defender_exploitability_calculator = self._DefenderExploitabilityCalculator(self)

    def export_settings(self, filename):
        pickle.dump((self.n_slots, self.n_types, self.prior, self.n_rounds, self.zero_sum, self.seed),
                    open(filename, "wb"))

    def export_payoff(self, filename):
        with open(filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for t in range(self.n_types):
                data = []
                for i in range(self.n_slots):
                    data.append(self.dfd_rew[i])
                for i in range(self.n_slots):
                    data.append(self.dfd_pen[i])
                for i in range(self.n_slots):
                    data.append(self.atk_rew[t, i])
                for i in range(self.n_slots):
                    data.append(self.atk_pen[t, i])
                writer.writerow(data)

    def get_lie_prob(self):
        # p00, p01, p10, p11 = self.gambit_solver.solve()
        # v0 = self.payoff[0, 0, 0, 0] * p10 + self.payoff[0, 0, 1, 0] * p11
        # v1 = self.payoff[0, 1, 0, 0] * p10 + self.payoff[0, 1, 1, 0] * p11
        # if np.isclose(v0, v1):
        #     return 0.
        # if v0 > v1:
        #     return p01
        # else:
        #     return p10

        self.gambit_solver.solve()
        p0 = self.gambit_solver.get_profile(0, [0])
        p1 = self.gambit_solver.get_profile(0, [])
        v = np.zeros(shape=self.n_slots, dtype=np.float32)
        for i in range(self.n_slots):
            for j in range(self.n_slots):
                v[i] += p1[j] * self.payoff[0, i, j, 0]

        optim = np.argmax(v)
        return 1. - p0[optim]

    def _get_base_ob(self):
        return self.ac_history.reshape(-1)

    def _get_dfd_ob(self, base_ob):
        return np.concatenate(([0.], base_ob))

    def _get_atk_ob(self, base_ob):
        return np.concatenate((self.type_ob, base_ob))

    def _get_ob(self):
        base_ob = self._get_base_ob()
        return [self._get_atk_ob(base_ob), self._get_dfd_ob(base_ob)]

    def get_ob_namers(self):
        def get_history_name(ob):
            if ob.shape[0] > 0:
                if ob[self.n_slots] < .5:
                    ac0 = self.n_slots
                    for i in range(self.n_slots):
                        if ob[i] > .5:
                            ac0 = i
                            break
                    ac1 = self.n_slots
                    for i in range(self.n_slots):
                        if ob[i + self.n_slots + 1] > .5:
                            ac1 = i
                            break
                    return ":({},{})".format(ac0, ac1) + get_history_name(ob[2 * (self.n_slots + 1):])
            return ""

        def atk_ob_namer(ob):
            name = ""
            for i in range(self.n_types):
                if ob[i] > .5:
                    name = str(i)
                    break
            return name + get_history_name(ob[self.n_types:])

        def dfd_ob_namer(ob):
            return "?" + get_history_name(ob[1:])

        return [atk_ob_namer, dfd_ob_namer]

    def reset(self, debug=False):
        self.rounds_so_far = 0
        self.ac_history = np.zeros(shape=self.ob_shape, dtype=np.float32)
        for r in range(self.n_rounds - 1):
            for p in range(2):
                self.ac_history[r][p][self.n_slots] = 1.
        self.atk_type = np.random.choice(self.n_types, p=self.prior)
        self.type_ob = np.zeros(shape=self.n_types, dtype=np.float32)
        self.type_ob[self.atk_type] = 1.
        return self._get_ob()

    def step(self, actions):
        # if actions[0] == actions[1]:
        #     atk_rew = self.atk_pen[self.atk_type][actions[0]]
        #     dfd_rew = self.dfd_rew[actions[1]]
        # else:
        #     atk_rew = self.atk_rew[self.atk_type][actions[0]]
        #     dfd_rew = self.dfd_pen[actions[1]]

        # assert np.isclose(atk_rew, self.payoff[self.atk_type, actions[0], actions[1], 0])
        # assert np.isclose(dfd_rew, self.payoff[self.atk_type, actions[0], actions[1], 1])

        atk_rew = self.payoff[self.atk_type, actions[0], actions[1], 0]
        dfd_rew = self.payoff[self.atk_type, actions[0], actions[1], 1]

        if self.rounds_so_far < self.n_rounds - 1:
            self.ac_history[self.rounds_so_far][0][self.n_slots] = 0.
            self.ac_history[self.rounds_so_far][0][actions[0]] = 1.
            self.ac_history[self.rounds_so_far][1][self.n_slots] = 0.
            self.ac_history[self.rounds_so_far][1][actions[1]] = 1.

        self.rounds_so_far += 1

        return self._get_ob(), [atk_rew, dfd_rew], [self.atk_type, self.atk_type], self.rounds_so_far >= self.n_rounds

    def calc_exploitability(self, i, strategy):
        if i == 0:
            return self._calc_attacker_exploitability(strategy)
        else:
            return self._calc_defender_exploitability(strategy)

    def _calc_attacker_exploitability(self, attacker_strategy):
        return self.attacker_exploitability_calculator.run(attacker_strategy)

    def _calc_defender_exploitability(self, defender_strategy):
        return self.defender_exploitability_calculator.run(defender_strategy)

    class _AttackerExploitabilityCalculator(object):
        def __init__(self, env):
            self.cache = None
            self.strategy = None
            self.n_slots = env.n_slots
            self.n_types = env.n_types
            self.n_rounds = env.n_rounds
            self.prior = env.prior
            self.payoff = env.payoff

        def _get_def_payoff(self, atk_ac, def_ac, prob):
            ret = 0.
            for t in range(self.n_types):
                ret += prob[t] * self.payoff[t, atk_ac, def_ac, 1]
            return ret

        def _reset(self):
            self.cache = dict()

        def _convert_to_type_ob(self, t):
            ob = np.zeros(shape=self.n_slots)
            ob[t] = 1.0
            return ob

        def _convert_to_atk_ob(self, history, t):
            ob = np.zeros(shape=(self.n_rounds - 1, 2, self.n_slots + 1))
            r = len(history)
            # print(r)
            for i in range(r):
                ob[i][0][history[i][0]] = 1.0
                ob[i][1][history[i][1]] = 1.0
            for i in range(r, self.n_rounds - 1):
                ob[i][0][self.n_slots] = 1.0
                ob[i][1][self.n_slots] = 1.0
            ob = np.concatenate([self._convert_to_type_ob(t), ob.reshape(-1)])
            return ob

        def _recursive(self, history, prior):
            if len(history) >= self.n_rounds:
                return 0.0
            if str(history) in self.cache:
                return self.cache[str(history)]
            else:
                atk_strategy_type = np.zeros(shape=(self.n_slots, self.n_types))
                for t in range(self.n_types):
                    atk_ob = self._convert_to_atk_ob(history, t)
                    atk_strategy = self.strategy(atk_ob)
                    for i in range(self.n_slots):
                        atk_strategy_type[i][t] += atk_strategy[i] * prior[t]

                max_ret = -1e100
                for def_ac in range(self.n_slots):
                    ret = 0.
                    for atk_ac in range(self.n_slots):
                        p = np.sum(atk_strategy_type[atk_ac])
                        prob = atk_strategy_type[atk_ac] / p
                        if p < 1e-5:
                            continue
                        r = self._get_def_payoff(atk_ac, def_ac, prob) + \
                            self._recursive(history + [[atk_ac, def_ac]], prob)
                        ret += r * p
                    max_ret = max(max_ret, ret)
                self.cache[str(history)] = max_ret
                return max_ret

        def run(self, attacker_strategy):
            self._reset()
            self.strategy = attacker_strategy
            return self._recursive([], self.prior)

    class _DefenderExploitabilityCalculator(object):
        def __init__(self, env):
            self.cache = None
            self.strategy = None
            self.n_slots = env.n_slots
            self.n_types = env.n_types
            self.n_rounds = env.n_rounds
            self.prior = env.prior
            self.payoff = env.payoff

        def _get_atk_payoff(self, t, atk_ac, def_ac):
            return self.payoff[t, atk_ac, def_ac, 0]

        def _reset(self):
            self.cache = dict()

        def _convert_to_def_ob(self, history):
            ob = np.zeros(shape=(self.n_rounds - 1, 2, self.n_slots + 1))
            r = len(history)
            # print(r)
            for i in range(r):
                ob[i][0][history[i][0]] = 1.0
                ob[i][1][history[i][1]] = 1.0
            for i in range(r, self.n_rounds - 1):
                ob[i][0][self.n_slots] = 1.0
                ob[i][1][self.n_slots] = 1.0
            ob = np.concatenate([[0.], ob.reshape(-1)])
            return ob

        def _recursive(self, history, t):
            if len(history) >= self.n_rounds:
                return 0.0
            if str(history) in self.cache:
                return self.cache[str(history)]
            else:
                def_ob = self._convert_to_def_ob(history)
                def_strategy = self.strategy(def_ob)

                max_ret = -1e100
                for atk_ac in range(self.n_slots):
                    ret = 0.
                    for def_ac in range(self.n_slots):
                        p = def_strategy[def_ac]
                        r = self._get_atk_payoff(t, atk_ac, def_ac) + \
                            self._recursive(history + [[atk_ac, def_ac]], t)
                        ret += r * p
                    max_ret = max(max_ret, ret)
                self.cache[str(history)] = max_ret
                return max_ret

        def run(self, defender_strategy):
            self.strategy = defender_strategy
            ret = 0.
            for t in range(self.n_types):
                self._reset()
                v = self._recursive([], t)
                ret += self.prior[t] * v
            return ret


def import_security_env(filename):
    n_slots, n_types, prior, n_rounds, zero_sum, seed = pickle.load(open(filename, "rb"))
    return SecurityEnv(n_slots=n_slots, n_types=n_types, prior=prior, n_rounds=n_rounds, zero_sum=zero_sum, seed=seed)
