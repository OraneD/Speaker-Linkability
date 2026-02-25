import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import glob
import os
from matplotlib.ticker import FuncFormatter

def load_score_dictionary(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def load_score_model(lst_path):
    score_lst = []
    for file in sorted(lst_path):
        print(f"Loading scores for {file}")
        score_lst.append(load_score_dictionary(file))
    # if len(score_lst) >= 3:
    #     score_lst[1], score_lst[2] = score_lst[2], score_lst[1]
    return score_lst


def compute_mean_scores(scores_dict):
    mean_scores = {}
    for speaker, dict_score in scores_dict.items():
        all_scores = np.concatenate([np.array(scores) for scores in dict_score.values()])
        mean_scores[speaker] = np.mean(all_scores)
    return mean_scores

def save_easy_speakers(mean_scores_dict, model_name, L_value, output_dir="speaker_linkability/easy_speakers"):
    os.makedirs(output_dir, exist_ok=True)

    scores = np.array(list(mean_scores_dict.values()))
    Q3 = np.percentile(scores, 75)

    easy_speakers = [spk for spk, mean in mean_scores_dict.items() if mean >= Q3]

    filename = f"{output_dir}/easy_speakers_{model_name.replace(' ', '_')}_L{L_value}.txt"

    with open(filename, "w") as f:
        f.write(f"(Q3) = {Q3:.4f}\n")
        f.write(f"Nb speakers : {len(mean_scores_dict)}\n")
        f.write(f"Nb worst speajers(>= Q3) : {len(easy_speakers)}\n")
        for spk in easy_speakers:
            f.write(f"{spk}\n")

    print(f"{filename} saved")
    return Q3, easy_speakers

def save_hard_speakers(mean_scores_dict, model_name, L_value, output_dir="speaker_linkability/hard_speakers"):
    """Sauvegarde les locuteurs <= Q1."""
    os.makedirs(output_dir, exist_ok=True)

    scores = np.array(list(mean_scores_dict.values()))
    Q1 = np.percentile(scores, 25)

    hard_speakers = [spk for spk, mean in mean_scores_dict.items() if mean <= Q1]

    filename = f"{output_dir}/hard_speakers_{model_name.replace(' ', '_')}_L{L_value}.txt"

    with open(filename, "w") as f:
        f.write(f"(Q1) = {Q1:.4f}\n")
        f.write(f"Nb speakers : {len(mean_scores_dict)}\n")
        f.write(f"Nb best speakers (<= Q1) : {len(hard_speakers)}\n")
        for spk in hard_speakers:
            f.write(f"{spk}\n")

    print(f"{filename} saved")
    return Q1, hard_speakers


def xtick_formatter(x, pos):
    if x == 0 or x == 1:
        return str(int(x))   
    return f"{x:.2f}"        
def plot_distribution(all_models):

    fig, axs = plt.subplots(3, 6, figsize=(24, 12), sharey=True)

    L_values = [1, 3, 5]
    model_names = ["ECAPA B3", "ECAPA B5", "WavLM ECAPA B3", "WavLM ECAPA B5", "ResNet B3", "ResNet B5"]
    colors = [ "#6a0dad", "blue", "#6a0dad", "blue", "#6a0dad", "blue"]
    hatches = ["", "", "//", "//", "XX", "XX"]

    for col, (model_name, model_scores, color, hatch) in enumerate(zip(model_names, all_models, colors, hatches)):
        for row, (scores_dict, L) in enumerate(zip(model_scores, L_values)):
            ax = axs[row, col]

            mean_scores_dict = compute_mean_scores(scores_dict)
            scores = list(mean_scores_dict.values())
            Q3, _ = save_easy_speakers(mean_scores_dict, model_name, L)
            Q1, _ = save_hard_speakers(mean_scores_dict, model_name, L)
            ax.hist(scores,
                    bins=np.arange(0, 1.01, 0.05),
                    weights=np.ones(len(scores)) / len(scores) * 100,
                    color=color, alpha=0.7, edgecolor="white", hatch=hatch)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 80)
            ax.grid(alpha=0.3)
            ax.axvline(Q1, color="springgreen", linestyle="--", linewidth=1.5, label="Q1",clip_on=False, zorder=5)
            ax.axvline(Q3, color="red",   linestyle="--", linewidth=1.5, label="Q3",clip_on=False, zorder=5)


        
            if row != len(L_values) - 1:
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(xtick_formatter))
                ax.tick_params(axis='x', labelsize=18)

            if col == 0:
                ax.set_ylabel(f"L={L}\n% speakers", fontsize=18)

            if row == 0:
                ax.set_title(model_name, fontsize=15)

            if row == len(L_values) - 1:
                ax.set_xlabel("Linkability", fontsize=18) 
            ax.tick_params(axis='y', labelsize=18, length=6, width=1.5)


    plt.tight_layout()
    plt.savefig("distribution_speakers.png", dpi=300)
    plt.savefig("distribution_speakers.pdf", dpi=300)

def main():
    speechbrain_B5 = load_score_model(glob.glob("../experiment/speaker_linkability/libri_SpeechBrain_ECAPA_B5/*.pkl"))
    speechbrain_B3 = load_score_model(glob.glob("../experiment/speaker_linkability/libri_SpeechBrain_ECAPA_B3/*.pkl"))
    ecapa_B5 = load_score_model(glob.glob("../experiment/speaker_linkability/libri_ECAPA_B5/*.pkl"))
    ecapa_B3 = load_score_model(glob.glob("../experiment/speaker_linkability/libri_ECAPA_B3/*.pkl"))
    resnet_B5 = load_score_model(glob.glob("../experiment/speaker_linkability/libri_resnet_B5/*.pkl"))
    resnet_B3 = load_score_model(glob.glob("../experiment/speaker_linkability/libri_resnet_B3/*.pkl"))


    all_models = [speechbrain_B3, speechbrain_B5, ecapa_B3, ecapa_B5, resnet_B3, resnet_B5]
    plot_distribution(all_models)


if __name__ == "__main__":
    main()