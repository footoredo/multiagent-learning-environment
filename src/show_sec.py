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
    for i in range(T):
        merged_result["episode"].append(result["episode"][i * 2])
        merged_result["exploitability"].append((result["exploitability"][i * 2] +
                                                result["exploitability"][i * 2 + 1] - 3.205 - 2.226) / 2)
    return merged_result


if __name__ == "__main__":
    folder = "../result/"
    res = pickle.load(open(folder + "security_seed:5410_gs_no-reset_5e-6_wolf_adv:20.0_1:1_latest.obj", "rb"))
    res = merge_exploitability(res)
    df = pd.DataFrame(data=res)
    sns.set()
    sns.lineplot(x="episode", y="exploitability", data=df)
    plt.show()
