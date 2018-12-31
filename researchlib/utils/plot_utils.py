import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

def plot_utils(s, u=None):
    plt.xscale('log')
    if u:
        plt.plot(u, s)
    else:
        plt.plot(s)
    plt.show()