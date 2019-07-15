import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from common.path_utils import *


def merge_exploitability(result):
    merged_result = {
        "episode": [],
        "exploitability": []
    }

    T = len(result["episode"]) // 2
    min0 = 1e100
    min1 = 1e100
    for i in range(T):
        merged_result["episode"].append(result["episode"][i * 2])
        min0 = min(min0, result["exploitability"][i * 2])
        min1 = min(min1, result["exploitability"][i * 2 + 1])
        merged_result["exploitability"].append((min0 + min1 - 3.2052 - 1.1333) / 2)
    return merged_result


if __name__ == "__main__":
    folder = "../result/"
    exp_name = "security_seed:5410_game:2-2-10_gs_no-reset_lr:5e-6_wolf_adv:20.0_latest_every:10_no-explore"
    exp_dir = join_path(folder, exp_name)
    # exp_dir = '.'
    res = joblib.load(join_path(exp_dir, "result.obj"))

    l = len(res["episode"])
    for i in range(l - 4, l):
        print(res["episode"][i], res["assessment"][i], res["player"][i])

    # res = merge_exploitability(res)
    df = pd.DataFrame(data=res)
    sns.set()
    # print(res)
    sns.lineplot(x="episode", y="assessment", hue="player", data=df)
    # g.set(yscale="log")
    # plt.interactive(False)
    plt.show()
