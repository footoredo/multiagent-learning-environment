import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def show_result(res, data_field, hue=None):
    df = pd.DataFrame(data=res)
    sns.set()
    if hue is None:
        sns.lineplot(x="episode", y=data_field, data=df)
    else:
        sns.lineplot(x="episode", y=data_field, hue=hue, data=df)
    plt.show()


def merge_results(results, sources, data_field):
    merged_result = {
        "episode": [],
        data_field: [],
        "source": []
    }
    for j, res in enumerate(results):
        length = len(res["episode"])
        for i in range(length):
            merged_result["episode"].append(res["episode"][i])
            merged_result["source"].append(sources[j])
            merged_result[data_field].append(res[data_field][i])
    return merged_result


def show_merged(results, sources, data_field):
    merged_result = merge_results(results, sources, data_field)
    show_result(merged_result, data_field=data_field, hue="source")


def get_exploitability(res):
    merged_res = {
        "episode": [],
        "exploitability": []
    }
    length = len(res["episode"]) // 2
    # print(length)
    for i in range(length):
        merged_res["episode"].append(res["episode"][i * 2])
        merged_res["exploitability"].append(
            (res["regret"][i * 2] + res["regret"][i * 2 + 1]) / 2
        )
    return merged_res


def show_exploitability(res):
    merged_res = get_exploitability(res)
    show_result(merged_res, data_field="exploitability")


def show_regret(res):
    show_result(res, data_field="regret", hue="player")


def show_prob(res):
    show_result(res, data_field="prob", hue="player")


if __name__ == "__main__":
    result0 = pickle.load(open("wolf_adv_5e-4_global.obj", "rb"))
    result1 = pickle.load(open("constant_5e-4_global.obj", "rb"))
    result2 = pickle.load(open("constant_2e-3_global.obj", "rb"))
    result0 = get_exploitability(result0)
    result1 = get_exploitability(result1)
    result2 = get_exploitability(result2)
    show_merged([result0, result1, result2],
                sources=["wolf_5e-4_4x", "constant_5e-4", "constant_2e-3"],
                data_field="exploitability")

