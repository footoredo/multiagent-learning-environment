from env.belief_security_env_vn import BeliefSecurityEnv
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run security game.")

    parser.add_argument('--seed', type=str)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--n-slots', type=int, default=2)
    parser.add_argument('--n-types', type=int, default=2)
    parser.add_argument('--n-rounds', type=int, default=2)
    # parser.add_argument('--prior', type=float, nargs='+')

    return parser.parse_args()


class Strategy(object):
    def __init__(self, strategy_fn, n):
        self.strategy_fn = strategy_fn
        self.n = n

    def strategy(self, ob):
        return self.strategy_fn(ob)

    def act(self, ob):
        s = self.strategy_fn(ob)
        return np.random.choice(range(self.n), p=s)

    def prob(self, ob, a):
        s = self.strategy_fn(ob)
        return s[a]


if __name__ == "__main__":
    args = parse_args()
    prior = np.array([1.0, 0.0])
    env = BeliefSecurityEnv(n_slots=args.n_slots, n_types=args.n_types, prior=prior, n_rounds=args.n_rounds,
                            seed=args.seed, export_gambit=False, beta=args.beta, random_prior=False)
    s = "0.4736	0.9882	0.4795"
    s = list(map(float, s.split()))

    def atk_strategy(ob):
        tp_ob = ob[:args.n_types]
        tp = None
        for i in range(args.n_types):
            if tp_ob[i] > 0.5:
                tp = i
                break
        if tp == 0:
            return np.array([s[0], 1-s[0]])
        else:
            return np.array([s[1], 1-s[1]])

    def dfd_strategy(ob):
        return np.array([s[2], 1-s[2]])

    env.assess_strategies((Strategy(atk_strategy, args.n_slots), Strategy(dfd_strategy, args.n_slots)))