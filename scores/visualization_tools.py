"""
The plot generated are designed to work with 3 matrices
Each function takes as argument a list of one of Scores() attribute
    - First function takes a list of Scores.all_score attribute
    - Second takes a list of Scores.speaker_scores attribute
"""
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

def plot_boxplot_linkability(speaker_dictionaries, fig_name):
    data = [np.array(list(speaker_dictionary.values())) for speaker_dictionary in speaker_dictionaries]
    box_colors = ["green", "red", "black"]
    plt.figure(figsize=(10, 5))
    boxplot = plt.boxplot(data, patch_artist=True, notch=True,vert=False, labels=("L=1", "L=3", "L=30"))

    for patch, color in zip(boxplot['boxes'], box_colors):
            patch.set_facecolor(color)

    for whisker in boxplot['whiskers']:
        whisker.set(color="black",
                    linewidth=1.5,
                    linestyle=":")

    for median in boxplot['medians']:
        median.set(color="grey", linewidth=2)
    
    for flier in boxplot['fliers']:
         flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
         
    for cap in boxplot['caps']:
        cap.set(color ='black',
                linewidth = 2)

    plt.xticks(np.arange(-0.2, 1.1, step=0.2))
    plt.xlabel("Linkability")
    plt.savefig(f"img/{fig_name}")
