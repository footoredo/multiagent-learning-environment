import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    res = pickle.load(open("sec_exp.obj", "rb"))
    print(res)
    df = pd.DataFrame(data=res)
    sns.set()
    sns.lineplot(x="episode", y="exploitability", data=df)
    plt.show()
