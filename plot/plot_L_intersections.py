import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def get_speakers_list(file, skip_lines=3):
    with open(file, "r") as f:
        lines = f.readlines()
    speakers = set(line.strip() for line in lines[skip_lines:] if line.strip())
    return speakers

def compute_intersection(list_of_sets):
    if not list_of_sets:
        return set()
    return set.intersection(*list_of_sets)

def compute_union(list_of_sets):
    if not list_of_sets:
        return set()
    return set.union(*list_of_sets)

def parse_filename(filename):
    # easy_speakers_ECAPA_B3_L1.txt -> arch=ECAPA, anonymizer=B3, L=L1
    base = os.path.basename(filename).replace(".txt", "")
    if "WavLM" in base.split("_") :
        arch = "WavLM_ECAPA" 
        anonymizer = base.split("_")[-2] 
        L = base.split("_")[-1]
    else : 
        parts = base.split("_")
        if len(parts) >= 5:
            _, _, arch, anonymizer, L = parts[:5]
        else:
            raise ValueError(f"Filename format unexpected: {filename}")
    return arch, anonymizer, L

def group_by_arch_anonymizer_L(files):
    """
    groups[(arch, anonymizer)][L] = set(speakers)
    """
    groups = {}
    for f in files:
        arch, anonymizer, L = parse_filename(f)
        spk = get_speakers_list(f)
        key = (arch, anonymizer)
        if key not in groups:
            groups[key] = {}
        groups[key][L] = spk
    return groups

def compute_L_pairwise_double(easy_groups, hard_groups, Ls=("L1","L3","L5")):

    L_pairs = [("L1","L3"), ("L1","L5"), ("L3","L5")]
    results = {}

    all_keys = set(easy_groups.keys()) | set(hard_groups.keys())
    for key in sorted(all_keys):
        arch, anonymizer = key

        easy_lv = easy_groups.get(key, {})
        hard_lv = hard_groups.get(key, {})

        res_list = []
        for a, b in L_pairs:
            if a in easy_lv and b in easy_lv:
                inter_e = compute_intersection([easy_lv[a], easy_lv[b]])
                union_e = compute_union([easy_lv[a], easy_lv[b]])
                jac_e = len(inter_e) / len(union_e) if union_e else 0.0
            else:
                inter_e = set()
                union_e = set()
                jac_e = 0.0

            if a in hard_lv and b in hard_lv:
                inter_h = compute_intersection([hard_lv[a], hard_lv[b]])
                union_h = compute_union([hard_lv[a], hard_lv[b]])
                jac_h = len(inter_h) / len(union_h) if union_h else 0.0
            else:
                inter_h = set()
                union_h = set()
                jac_h = 0.0

            label = f"{a}_vs_{b}"
            res_list.append(
                (label,
                 len(inter_e), len(union_e), jac_e,
                 len(inter_h), len(union_h), jac_h)
            )

        results[key] = res_list

    return results

def plot_L_comparisons_double(results):
    architectures = ["ECAPA", "ResNet", "WavLM_ECAPA"]
    anonymizers = ["B3", "B5"]
    L_pair_labels = ["L1_vs_L3", "L1_vs_L5", "L3_vs_L5"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 8), sharey=True)
    fig.suptitle("Intersection / Union L1 vs L3 vs L5", fontsize=15)

    handles = [
        plt.Rectangle((0,0),1,1,color="#ec6a37",label="Easy Inter"),
        plt.Rectangle((0,0),1,1,color="#d30404d8",label="Easy Union"),
        plt.Rectangle((0,0),1,1,color="#05b605",label="Hard Inter"),
        plt.Rectangle((0,0),1,1,color="#127712E4",label="Hard Union")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, 0.94), frameon=True)

    for i, arch in enumerate(architectures):
        for j, anonymizer in enumerate(anonymizers):
            ax = axs[i, j]

            key = (arch, anonymizer)
            res_list = results.get(key,
                [(lbl,0,0,0.0,0,0,0.0) for lbl in L_pair_labels])

            labels = [r[0] for r in res_list]
            Ei = [r[1] for r in res_list]
            Eu = [r[2] for r in res_list]
            Ej = [r[3] for r in res_list]
            Hi = [r[4] for r in res_list]
            Hu = [r[5] for r in res_list]
            Hj = [r[6] for r in res_list]

            x = np.arange(len(labels))
            width = 0.18
            offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

            ax.bar(x + offsets[0], Ei, width, color="#ec6a37")
            ax.bar(x + offsets[1], Eu, width, color="#d30404d8")
            ax.bar(x + offsets[2], Hi, width, color="#05b605")
            ax.bar(x + offsets[3], Hu, width, color="#127712E4")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_title(f"{arch} — {anonymizer}")
            if j == 0:
                ax.set_ylabel("speaker number")

            ymax = max(Eu + Hu) * 1.18 if (Eu+Hu) else 10
            ax.set_ylim(0, ymax + 1000)

            for k in range(len(labels)):
                top = max(Eu[k], Hu[k])
                ax.text(k, top + 0.015 * ymax,
                        f"J-easy={Ej[k]:.3f}\nJ-hard={Hj[k]:.3f}",
                        ha="center", fontsize=8)

            ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("plots/L_easy_hard.png", dpi=300)
    plt.show()

def main():
    easy_files = glob.glob("../experiment/speaker_linkability/easy_speakers/*.txt")
    hard_files = glob.glob("../experiment/speaker_linkability/hard_speakers/*.txt")

    easy_groups = group_by_arch_anonymizer_L(easy_files)
    hard_groups = group_by_arch_anonymizer_L(hard_files)

    results = compute_L_pairwise_double(easy_groups, hard_groups)

    plot_L_comparisons_double(results)

if __name__ == "__main__":
    main()
