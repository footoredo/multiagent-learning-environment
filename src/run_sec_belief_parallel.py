from env.matrix_env import MatrixEnv
from env.belief_security_env_vn import BeliefSecurityEnv
from controller.belief_controller import BeliefController
from agent.dummy_agent import DummyAgent
from agent.infant_agent import InfantAgent
from agent.self_play_agent import SelfPlayAgent
from agent.ppo_agent_backup import PPOAgent
from agent.ppo_agent_stacked_v3 import PPOAgentStacked
from agent.pac_agent import PACAgent
from agent.mlp_policy import MLPPolicy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from decimal import Decimal
from common.path_utils import *
import joblib
import argparse
import subprocess
import joblib



def parse_args():
    parser = argparse.ArgumentParser(description="Run security game.")

    parser.add_argument('--seed', type=str)
    parser.add_argument('--n-slots', type=int, default=2)
    parser.add_argument('--n-rounds', type=int, default=2)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--start-round', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":

    def argument_to_tuple(argument):
        if type(argument) == list and len(argument) == 1:
            return argument[0]
        elif type(argument) == str:
            return argument
        else:
            parameters = list(map(float, argument[1:]))
            return tuple(argument[:1] + parameters)

    args = parse_args()

    beta = args.beta
    seed = args.seed
    n_slots = args.n_slots
    # other = "1000-test-steps-large-network"

    result_folder = "../result/"
    save_dir = os.path.join(result_folder, "{}-{}-{}".format(seed, n_slots, beta))

    train = True

    def join_paths(paths):
        if len(paths) == 1:
            return paths[0]
        else:
            return os.path.join(paths[0], join_paths(paths[1:]))


    def train(p, r):
        if r == 1:
            vn_load_path = None
            sub_load_path = None
        else:
            vn_load_path = os.path.join(save_dir, "round-{}".format(r - 1))
            sub_load_path = save_dir

        arg = ["python", "run_sec_belief_part.py", "--beta={}".format(beta), "--seed={}".format(seed),
               "--n-slots={}".format(n_slots),
               "--n-types={}".format(2), "--n-rounds={}".format(r),
               "--prior", "{:.1f}".format(p / 10), "{:.1f}".format(1 - p / 10),
               "--save-dir={}".format(join_paths([save_dir, "train", "r-{}".format(r), "p-{}".format(p)]))]
        if r > 1:
            arg += ["--vn-load-path={}".format(vn_load_path),
                    "--sub-load-path={}".format(sub_load_path)]

        sp = subprocess.Popen(arg)
        return sp

    for r in range(args.start_round, args.n_rounds + 1):
        sps = []
        for p in range(5):
            sp = train(p, r)
            sps.append(sp)

        for sp in sps:
            sp.wait()

        sps = []
        for p in range(5, 11):
            sp = train(p, r)
            sps.append(sp)

        for sp in sps:
            sp.wait()

        for_sheet = []

        line = []
        line.append("")
        line.append("strategy")
        for i in range((1 if n_slots == 2 else n_slots) * 3 - 1):
            line.append("")
        line.append("Payoff")
        for i in range(2):
            line.append("")
        line.append("Distance")
        for i in range(2):
            line.append("")
        line.append("PBNE Distance")
        for i in range(2):
            line.append("")
        for_sheet.append("\t".join(line) + "\n")

        line = []
        line.append("n_rounds={}".format(r))
        line.append("Attacker-0")
        for i in range((1 if n_slots == 2 else n_slots) - 1):
            line.append("")
        line.append("Attacker-1")
        for i in range((1 if n_slots == 2 else n_slots) - 1):
            line.append("")
        line.append("Defender")
        for i in range((1 if n_slots == 2 else n_slots) - 1):
            line.append("")
        line.append("Attacker-0")
        line.append("Attacker-1")
        line.append("Defender")
        line.append("Attacker-0")
        line.append("Attacker-1")
        line.append("Defender")
        line.append("Attacker-0")
        line.append("Attacker-1")
        line.append("Defender")
        for_sheet.append("\t".join(line) + "\n")

        atk_payoff = [[] for _ in range(2)]
        dfd_payoff = []
        atk_strat = []
        dfd_strat = []

        for p in range(11):
            path = join_paths([save_dir, "train", "r-{}".format(r), "p-{}".format(p)])
            info = joblib.load(os.path.join(path, "part.info"))
            line = []

            def get_s(s):
                if n_slots == 2:
                    return [s[0]]
                else:
                    return s.tolist()

            line += sum([get_s(info["strategy"][0][t]) for t in range(2)], [])
            line += get_s(info["strategy"][1])
            line += info["payoff"][0]
            line.append(info["payoff"][1])
            line += info["distance"][0][0]
            line.append(info["distance"][1][0])
            line += info["distance"][0][1]
            line.append(info["distance"][1][1])
            for_sheet.append("{:.1f}\t".format(p / 10) + "\t".join(list(map(str, line))) + "\n")

            pp = np.array([p / 10, 1 - p / 10])
            for i in range(2):
                tp = np.zeros(2)
                tp[i] = 1.
                atk_payoff[i].append((pp, info["payoff"][0][i]))
                atk_strat.append((np.concatenate([tp, pp]), info["strategy"][0][i]))
            dfd_strat.append((pp, info["strategy"][1]))
            dfd_payoff.append((pp, info["payoff"][1]))

        joblib.dump(atk_strat, join_path_and_check(os.path.join(save_dir, "agent-0"), "pack-{}.obj".format(r)))
        joblib.dump(dfd_strat, join_path_and_check(os.path.join(save_dir, "agent-1"), "pack-{}.obj".format(r)))
        for i in range(2):
            joblib.dump(atk_payoff[i], join_path_and_check(os.path.join(save_dir, "round-{}".format(r)), "atk{}-vn.obj".format(i)))
        joblib.dump(dfd_payoff, join_path_and_check(os.path.join(save_dir, "round-{}".format(r)), "dfd-vn.obj"))

        # import pyperclip
        # pyperclip.copy("\n".join(for_sheet))
        with open(os.path.join(save_dir, "{}-for_sheet".format(r)), "w") as f:
            f.writelines(for_sheet)

else:
    print("fuck")