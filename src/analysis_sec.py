from env.security_env import SecurityEnv, import_security_env
from monitor.statistics import Statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from common.path_utils import *
import joblib
import numpy as np

exp_name = "security_seed:5410_game:2-2-5-0.5:0.5_gs_no-reset_5e-6_wolf_adv:20.0_latest_every:10_1000-test-steps-large-network"
step = 20000
exp_dir = join_path("../result", exp_name)
step_dir = join_path(exp_dir, "step-{}".format(step))
env = import_security_env(join_path(exp_dir, "env_settings.obj"))


def history_to_ob(history):
    h = np.zeros(shape=env.ob_shape)
    len_h = len(history)
    for i in range(len_h):
        h[i][history[i]] = 1.
    for i in range(len_h, h.shape[0]):
        h[i][env.n_slots] = 1.
    return h.reshape(-1)


def get_atk_ob(history):
    type_ob = np.zeros(shape=env.n_types)
    type_ob[history[0]] = 1.
    return np.concatenate([type_ob, history_to_ob(history[1:])])


def analysis_final_assessment(final_assessment, statistics):
    atk_result, def_result = final_assessment

    result = []
    for h, v in atk_result:
        result.append(v)

    result = sorted(result)
    print("95%:", result[int(0.95 * len(result))])
    print("99%:", result[int(0.99 * len(result))])
    print("99.9%:", result[int(0.999 * len(result))])

    # sns.distplot(result)
    # plt.show()

    standard = 0

    stupid_result = {
        "len": [],
        "freq": [],
        "eps": [],
        "type": []
    }
    for h, v in atk_result:
        if len(h) == 5 and h[0] == 0:
        # if True:
            if v > standard:
                stupid_result["len"].append(len(h) - 1)
                stupid_result["freq"].append(statistics.get_freq(0, get_atk_ob(h)))
                # stupid_result["eps"].append(v / (11 - len(h)))
                stupid_result["eps"].append(v)
                stupid_result["type"].append(h[0])

    df = pd.DataFrame(data=stupid_result)
    # sns.relplot(x="freq", y="eps", hue="type", data=df)
    # sns.jointplot(x="freq", y="eps", kind="hex", data=df)
    sns.jointplot(x="freq", y="eps", data=df)
    plt.show()


def analysis():
    final_assessment = joblib.load(join_path(exp_dir, "final_assessment.obj"))
    statistics = Statistics(env)
    statistics.load(join_path(step_dir, "statistics.obj"), full=False)
    # statistics.save(join_path(step_dir, "statistics-compact.obj"), full=False)

    analysis_final_assessment(final_assessment, statistics)


if __name__ == "__main__":
    analysis()
