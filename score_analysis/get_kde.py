import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pickle

speaker_m1 = pickle.load(open("experiment/scores/speaker_scores_matrix_L-1_Resnet_anon_B5.pkl", "rb"))
speaker_m3 = pickle.load(open("experiment/scores/speaker_scores_matrix_L-3_Resnet_anon_B5.pkl", "rb"))
speaker_m30 = pickle.load(open("experiment/scores/speaker_scores_matrix_L-30_Resnet_anon_B5.pkl", "rb"))

speaker_m1_ecapa = pickle.load(open("experiment/scores/speaker_scores_matrix_L-1_ECAPA_anon_B5.pkl", "rb"))
speaker_m3_ecapa = pickle.load(open("experiment/scores/speaker_scores_matrix_L-3_ECAPA_anon_B5.pkl", "rb"))
speaker_m30_ecapa = pickle.load(open("experiment/scores/speaker_scores_matrix_L-30_ECAPA_anon_B5.pkl", "rb"))

def make_kde(dic1, dic2, dic3, savename):
    scores1 = list(dic1.values())
    scores2 = list(dic2.values())
    scores3 = list(dic3.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    data = [(scores1, "L=1"),
            (scores2, "L=3"),
            (scores3, "L=30")]

    colors = ['green', 'red', 'black']

    for ax, (scores, title), color in zip(axes, data, colors):
        sns.histplot(scores, bins=30, kde=True, stat="density",
                    color=color, edgecolor='black', alpha=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.grid(True)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../img/{savename}")

make_kde(speaker_m1, speaker_m3, speaker_m30, "KDE_resnet_anon.png")
make_kde(speaker_m1_ecapa, speaker_m3_ecapa, speaker_m30_ecapa, "KDE_ECAPA_anon_B5.png")
