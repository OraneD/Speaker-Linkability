import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import glob

def load_score_dictionary(path):
    with open(path, "rb") as f:
        dico = pkl.load(f)
        return dict(sorted(dico.items(), key=lambda x: int(x[0])))

def load_score_model(lst_path):
    score_lst = []
    for file in sorted(lst_path):
        print(f"Loading scores for {file}")
        score_lst.append(load_score_dictionary(file))
    # Be cautious with the order given by glob here
    # if len(score_lst) >= 3:
    #     score_lst[1], score_lst[2] = score_lst[2], score_lst[1]
    return score_lst

def compute_average(score_dict):
    """
    Takes a dict {N: [scores]} et return:
    - x : np.array on N
    - y : np.array of means
    - std : np.array of stds
    """
    x = np.array([int(k) for k in score_dict.keys()])
    y = np.array([np.mean(v) for v in score_dict.values()])
    std = np.array([np.std(v, ddof=1) for v in score_dict.values()])
    return x, y, std

def plot_scores(lst_dictionary):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    labels = ["L = 1", "L = 3", "L = 5"]

    styles = {
        0: ("blue", "o", "WavLM_ECAPA B5"),
        1: ("purple", "o", "WavLM_ECAPA B3"),
        2: ("black", "o", "WavLM_ECAPA orig"),
        3: ("blue", "D", "ResNet B5"),
        4: ("purple", "D", "ResNet B3"),
        5: ("black", "D", "ResNet orig"),
        6: ("blue", "X", "ECAPA B5"),
        7: ("purple", "X", "ECAPA B3"),
        8: ("black", "X", "ECAPA orig"),
    }

    for i, ax in enumerate(axs):
        ax.set_title(labels[i])

        for model_idx, scores in enumerate(lst_dictionary):
            score = scores[i]  
            color, marker, label = styles[model_idx]

            x, y, std = compute_average(score)

            ax.errorbar(
                x, y,
                #yerr=std,
                linestyle="dotted",
                marker=marker,
                markersize=5,
                color=color,
                elinewidth=1,      
                label=label
            )

        ax.set_xlabel("Number of enrollment speakers")
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000, 10000])
        ax.set_xticklabels(["$10^1$", "$10^2$", "$10^3$", "$10^4$"])
        ax.set_yticks(np.arange(0, 1.1, step=0.2))

        N_values = np.array([int(k) for k in lst_dictionary[0][i].keys()])
        chance = 1 / N_values
        ax.plot(N_values, chance, linestyle="--", color="gray", label="Chance level")

    axs[0].set_ylabel("Linkability")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
                          ncol=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("linkability_3attackers.png", dpi=300)
    plt.show()

def main():
    ecapa_libri_B5 = load_score_model(glob.glob("../experiment/overall_linkability/libri_ECAPA_B5/*.pkl"))
    ecapa_libri_B3 = load_score_model(glob.glob("../experiment/overall_linkability/libri_ECAPA_B3/*.pkl"))
    ecapa_orig = load_score_model(glob.glob("../experiment/overall_linkability/libri_ECAPA_orig/*.pkl"))
    resnet_libri_B5 = load_score_model(glob.glob("../experiment/overall_linkability/libri_resnet_B5/*.pkl"))
    resnet_libri_B3 = load_score_model(glob.glob("../experiment/overall_linkability/libri_resnet_B3/*.pkl"))
    resnet_orig = load_score_model(glob.glob("../experiment/overall_linkability/libri_resnet_orig/*.pkl"))
    speechbrain_B5 = load_score_model(glob.glob("../experiment/overall_linkability/libri_SpeechBrain_ECAPA_B5/*.pkl"))
    speechbrain_B3 = load_score_model(glob.glob("../experiment/overall_linkability/libri_SpeechBrain_ECAPA_B3/*.pkl"))
    speechbrain_orig = load_score_model(glob.glob("../experiment/overall_linkability/libri_SpeechBrain_ECAPA_orig/*.pkl"))

    plot_scores([ecapa_libri_B5,
                 ecapa_libri_B3, 
                 ecapa_orig, 
                 resnet_libri_B5, 
                 resnet_libri_B3, 
                 resnet_orig, 
                 speechbrain_B5, 
                 speechbrain_B3, 
                 speechbrain_orig])

main()
