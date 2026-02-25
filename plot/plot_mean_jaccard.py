import numpy as np
import matplotlib.pyplot as plt

"""
Mean jaccard similarities plot 
The values have been extracted from the plot given by plot_anonymizer_intersections.py, plot_architecture_intersections.py, plot_L_intersections.py
"""

B3_vs_B5 = [(0.211, 0.321), (0.236, 0.315), (0.44, 0.319), (0.281, 0.350), (0.287,0.362), (0.318,0.381), (0.271, 0.328), (0.289, 0.353), (0.269, 0.368)]
ecapa_vs_resnet_vs_WavLM = [(0.429,0.513), (0.393, 0.527), (0.415, 0.543), (0.350, 0.455), (0.360, 0.384), (0.367, 0.430), #WavLM vs ResNet
                            (0.425, 0.494), (0.418, 0.497),  (0.428, 0.543), (0.320, 0.533), (0.338, 0.405), (0.347, 0.406), #ECAPA vs ResNet
                            (0.418, 0.480), (0.401, 0.502), (0.430, 0.534), (0.311, 0.434),  (0.322, 0.364), (0.329, 0.391) #ECAPA vs WavLM
                            ]

L1_vs_L3 = [(0.253, 0.332), (0.280, 0.332), (0.249, 0.325), (0.254, 0.299), (0.245, 0.317), (0.232, 0.411)]
L1_vs_L5 = [(0.288, 0.343), (0.261, 0.344), (0.255, 0.313), (0.244, 0.323), (0.263, 0.329), (0.240, 0.354)]
L3_vs_L5 = [(0.334,0.416), (0.324, 0.401), (0.295, 0.348), (0.286, 0.352), (0.307, 0.405), (0.287, 0.387)]

L_all = L1_vs_L3 + L1_vs_L5 + L3_vs_L5

data = {
    "Anonymizer": B3_vs_B5,
    "Attacker": ecapa_vs_resnet_vs_WavLM,
    "Conversation length (L)": L_all
}

easy_means, easy_stds = [], []
hard_means, hard_stds = [], []
labels = []

for name, lst in data.items():
    arr = np.array(lst)
    easy_means.append(arr[:, 0].mean())
    easy_stds.append(arr[:, 0].std())
    hard_means.append(arr[:, 1].mean())
    hard_stds.append(arr[:, 1].std())
    labels.append(name)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 3.2))

bars_easy = ax.bar(
    x - width/2, easy_means, width,
    yerr=easy_stds, color="red", label='Easy-to-link speakers'
)
bars_hard = ax.bar(
    x + width/2, hard_means, width,
    yerr=hard_stds, color="green", label='Hard-to-link speakers'
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)

ax.set_ylabel("Mean Jaccard", fontsize=12)
ax.tick_params(axis='y', labelsize=11)

ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

def add_bar_labels(bars, stds):
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 0.01,  
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

all_heights = np.array(easy_means + hard_means)
all_stds = np.array(easy_stds + hard_stds)

y_max = (all_heights + all_stds).max()
ax.set_ylim(top=y_max + 0.05)  


add_bar_labels(bars_easy, easy_stds)
add_bar_labels(bars_hard, hard_stds)

plt.tight_layout()
plt.savefig("../img/mean_jaccard.pdf", dpi=300)
plt.show()
