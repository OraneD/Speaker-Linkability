import matplotlib.pyplot as plt
import numpy as np

def plot_overall_linkability(scores_dictionaries, fig_name):

    plt.figure(figsize=(10, 5))
    colors = ["green", "red", "black"]
    labels = ["L = 1", "L = 3", "L = 30"]


    for i, score in enumerate(scores_dictionaries) :
        x = np.array([x for x in score.keys()])
        y = np.array([x for x in score.values()])
        color = colors[i]
        label = labels[i]
        plt.plot(x,y, linestyle="dotted", marker="o",markersize=4, color=color, label=label)

    plt.xlabel("Number of enrollment speakers")
    plt.ylabel("Linkability")
    plt.xscale("log")  
    plt.xticks([10, 100, 1000, 10000], ["$10^1$", "$10^2$", "$10^3$", "$10^4$"])

    plt.yticks(np.arange(0, 1.1, step=0.2))
    plt.legend()
    plt.savefig(f"img/{fig_name}")

