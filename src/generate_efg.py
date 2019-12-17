from env.belief_security_env_vn import BeliefSecurityEnv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run security game.")

    parser.add_argument('--seed', type=str)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--n-slots', type=int, default=2)
    parser.add_argument('--n-types', type=int, default=2)
    parser.add_argument('--n-rounds', type=int, default=2)
    parser.add_argument('--prior', type=float, nargs='+')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = BeliefSecurityEnv(n_slots=args.n_slots, n_types=args.n_types, prior=args.prior, n_rounds=args.n_rounds,
                            seed=args.seed, export_gambit=True, beta=args.beta)
