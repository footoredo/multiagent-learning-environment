import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from common.path_utils import *

folder = "../result/"

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


def show_local_result(exp_name):
    exp_dir = join_path(folder, exp_name)
    # exp_dir = '.'
    # res = joblib.load(join_path(exp_dir, "result.obj"))
    loc_res = joblib.load(join_path(exp_dir, "local_results.obj"))

    loc_res_t = {
        "episode": [],
        "prob": [],
        "index": []
    }

    samples = [0]
    # print(loc_res)
    side = 0
    action = 1
    key_list = list(loc_res[0][side].keys())
    keys = [key_list[i] for i in samples]
    sums = [0.0 for _ in range(len(samples))]
    tots = [0 for _ in range(len(samples))]
    start = 0

    for i in range(start, len(loc_res)):
        for j in range(len(samples)):
            if keys[j] in loc_res[i][side]:
                loc_res_t["episode"].append(i + 1)
                loc_res_t["prob"].append(loc_res[i][side][keys[j]][action])
                loc_res_t["index"].append("atk-{}".format(j))
                tots[j] += 1
                sums[j] += loc_res[i][side][keys[j]][action]

                loc_res_t["episode"].append(i + 1)
                loc_res_t["prob"].append(sums[j] / tots[j])
                loc_res_t["index"].append("atk-{}-local-avg".format(j))

    # for j in range(len(samples)):
    #     for i in range(start, len(loc_res)):
    #         # loc_res_t["episode"].append(i + 1)
    #         # loc_res_t["prob"].append(sums[j] / tots[j])
    #         # loc_res_t["index"].append("{}-global-avg".format(j))
    #         loc_res_t["episode"].append(i + 1)
    #         loc_res_t["prob"].append(0.227)
    #         loc_res_t["index"].append("{}-std".format(j))

    samples = [0]
    side = 1
    key_list = list(loc_res[0][side].keys())
    keys = [key_list[i] for i in samples]
    sums = [0.0 for _ in range(len(samples))]
    tots = [0 for _ in range(len(samples))]
    start = 0

    for i in range(start, len(loc_res)):
        for j in range(len(samples)):
            if keys[j] in loc_res[i][side]:
                loc_res_t["episode"].append(i + 1)
                loc_res_t["prob"].append(loc_res[i][side][keys[j]][action])
                loc_res_t["index"].append("def-{}".format(j))
                tots[j] += 1
                sums[j] += loc_res[i][side][keys[j]][action]

                loc_res_t["episode"].append(i + 1)
                loc_res_t["prob"].append(sums[j] / tots[j])
                loc_res_t["index"].append("def-{}-local-avg".format(j))

    # l = len(res["episode"])
    # for i in range(l - 4, l):
    #     print(res["episode"][i], res["assessment"][i], res["player"][i])

    # res = merge_exploitability(res)
    # df = pd.DataFrame(data=res)
    df = pd.DataFrame(data=loc_res_t)
    sns.set()
    # print(res)
    # sns.lineplot(x="episode", y="assessment", hue="player", data=df)
    sns.lineplot(x="episode", y="prob", hue="index", data=df)
    # g.set(yscale="log")
    # plt.interactive(False)
    plt.show()


def show_result(exp_name):
    exp_dir = join_path(folder, exp_name)
    # exp_dir = '.'
    res = joblib.load(join_path(exp_dir, "result.obj"))

    l = len(res["episode"])
    for i in range(l - 4, l):
        print(res["episode"][i], res["assessment"][i], res["player"][i])

    # res = merge_exploitability(res)
    df = pd.DataFrame(data=res)
    # df = pd.DataFrame(data=loc_res_t)
    sns.set()
    # print(res)
    sns.lineplot(x="episode", y="assessment", hue="player", data=df)
    # sns.lineplot(x="episode", y="prob", hue="index", data=df)
    # g.set(yscale="log")
    # plt.interactive(False)
    plt.show()


if __name__ == "__main__":
    # exp_name = "security_ppo_seed:5410_game:2-2-5-0.5:0.5_gs_no-reset_5e-6_wolf_adv:20.0_latest_test_every:10_test_steps:1000_network:256-4_train:10*10"
    # exp_name = "security_ppo_seed:5410_game:2-2-5-0.5:0.5_gs_no-reset_5e-6_constant_latest_test_every:10_test_steps:1000_network:256-4_train:10*10"
    # exp_name = "security_ppo_seed:5411_game:2-2-5-0.5:0.5_gs_no-reset_5e-6_constant_latest_test_every:10_test_steps:1000_network:256-4_train:10*10"
    # exp_name = "security_ppo_seed:5410_game:2-2-2-0.5:0.5_gs_no-reset_5e-6_constant_latest_test_every:5_test_steps:1000_network:256-4_train:4*10"
    # exp_name = "security_ppo_seed:5410_game:2-2-2-0.5:0.5_gs_no-reset_5e-6_wolf_adv:20.0_latest_test_every:5_test_steps:1000_network:256-4_train:4*10"
    # exp_name = "security_seed:5410_game:3-2-2-0.5:0.5_gs_no-reset_5e-6_constant_latest_test_every:10_test_steps:1000_network:256-4_train:20*5"
    # exp_name = "security_ppo_seed:5410_game:3-2-2-0.5:0.5_gs_no-reset_5e-6_constant_latest_test_every:10_test_steps:1000_network:256-4_train:4*10"
    exp_name = "security_ppo_seed:5410_game:5-2-5-0.5:0.5_gs_no-reset_5e-6_constant_latest_test_every:10_test_steps:1000_network:256-4_train:10*20"
    show_result(exp_name)
    # show_local_result(exp_name)