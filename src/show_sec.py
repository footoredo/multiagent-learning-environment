import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
        merged_result["exploitability"].append((min0 + min1 - 12.85014436 - 2.75608594) / 2)
    return merged_result


if __name__ == "__main__":
    folder = "../result/"
    res = pickle.load(open(folder + "security_seed:5410_2-2-8_gs_no-reset_5e-6_wolf_adv:20.0_1:1_latest_10-10000.obj", "rb"))
    res = merge_exploitability(res)
    df = pd.DataFrame(data=res)
    sns.set()
    g = sns.lineplot(x="episode", y="exploitability", data=df)
    # g.set(yscale="log")
    plt.show()
