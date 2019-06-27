import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    result = pickle.load(open("wolf.obj", "rb"))
    l = len(result["episode"])
    for i in range(l):
        result["player"][i] = result["player"][i][:8]

    df = pd.DataFrame(data=result)
    sns.set()
    sns.relplot(x="episode", y="prob", hue="player", kind="line", data=df)
    plt.show()