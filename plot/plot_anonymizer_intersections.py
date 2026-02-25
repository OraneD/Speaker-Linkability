import glob
import os
import matplotlib.pyplot as plt
import numpy as np



def get_speakers_list(file, skip_lines=3):
    with open(file, "r") as f:
        lines = f.readlines()
    speakers = set(line.strip() for line in lines[skip_lines:] if line.strip())
    return speakers

def parse_filename(filename):
    """
    easy_speakers_ECAPA_B3_L1.txt → arch, anonymizer, L
    """
    base = os.path.basename(filename).replace(".txt", "")
    if "WavLM" in base.split("_") :
        arch = "WavLM_ECAPA" 
        anonymizer = base.split("_")[-2] 
        L = base.split("_")[-1]
    else : 
        _, _, arch, anonymizer, L = base.split("_")
    return arch, anonymizer, L


def group_files_by_arch_L(files):
    """
    groups[(arch, L)][anonymizer] = set_speakers
    """
    groups = {}
    for f in files:
        arch, anonymizer, L = parse_filename(f)
        spk = get_speakers_list(f)

        key = (arch, L)
        if key not in groups:
            groups[key] = {}

        groups[key][anonymizer] = spk

    return groups


def build_global_anonymizers(groups):
    """
    ECAPA_GLOBAL / ResNet_GLOBAL by anonymizer (B3 et B5)
    """
    global_groups = {
        "ECAPA": {"B3": set(), "B5": set()},
        "ResNet": {"B3": set(), "B5": set()},
        "WavLM_ECAPA": {"B3": set(), "B5": set()}
    }

    for (arch, L), anonymizer_dict in groups.items():
        for anonymizer, speakers in anonymizer_dict.items():
            global_groups[arch][anonymizer] |= speakers

    return global_groups



def compare_anonymizers(groups_easy, groups_hard):
    results = []


    for key in sorted(set(groups_easy.keys()) | set(groups_hard.keys())):
        arch, L = key

        if key in groups_easy and "B3" in groups_easy[key] and "B5" in groups_easy[key]:
            s3e = groups_easy[key]["B3"]
            s5e = groups_easy[key]["B5"]
            inter_e = len(s3e & s5e)
            union_e = len(s3e | s5e)
            j_e = inter_e / union_e if union_e else 0
        else:
            inter_e = union_e = j_e = 0

        if key in groups_hard and "B3" in groups_hard[key] and "B5" in groups_hard[key]:
            s3h = groups_hard[key]["B3"]
            s5h = groups_hard[key]["B5"]
            inter_h = len(s3h & s5h)
            union_h = len(s3h | s5h)
            j_h = inter_h / union_h if union_h else 0
        else:
            inter_h = union_h = j_h = 0

        results.append(
            ((arch, L),
             inter_e, union_e, j_e,
             inter_h, union_h, j_h)
        )

    global_easy = build_global_anonymizers(groups_easy)
    global_hard = build_global_anonymizers(groups_hard)

    for arch in ["ECAPA", "ResNet", "WavLM_ECAPA"]:

        s3e = global_easy[arch]["B3"]
        s5e = global_easy[arch]["B5"]
        inter_e = len(s3e & s5e)
        union_e = len(s3e | s5e)
        j_e = inter_e / union_e if union_e else 0

        s3h = global_hard[arch]["B3"]
        s5h = global_hard[arch]["B5"]
        inter_h = len(s3h & s5h)
        union_h = len(s3h | s5h)
        j_h = inter_h / union_h if union_h else 0

        results.append(
            ((arch, "GLOBAL"),
             inter_e, union_e, j_e,
             inter_h, union_h, j_h)
        )

    plot_results(results)



def plot_results(results):

    labels = [f"{arch}-{lvl}" for (arch, lvl), *_ in results]

    easy_inter = [r[1] for r in results]
    easy_union = [r[2] for r in results]
    easy_j =    [r[3] for r in results]

    hard_inter = [r[4] for r in results]
    hard_union = [r[5] for r in results]
    hard_j =    [r[6] for r in results]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15, 8))

    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    ax.bar(x + offsets[0], easy_inter, width, label="Easy Intersection", color="#ec6a37")
    ax.bar(x + offsets[1], easy_union, width, label="Easy Union", color="#d30404d8")

    ax.bar(x + offsets[2], hard_inter, width, label="Hard Intersection", color="#05b605")
    ax.bar(x + offsets[3], hard_union, width, label="Hard Union", color="#127712E4")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("speaker number")

    fig.suptitle("Intersection / Union – B3 vs B5 ", fontsize=16)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4)

    y_max = max(max(easy_union), max(hard_union)) * 1.25
    ax.set_ylim(0, y_max)

    for k in range(len(labels)):
        ymax = max(easy_union[k], hard_union[k])
        ax.text(k, ymax + 0.03 * y_max,
                f"J-easy:{easy_j[k]:.3f}\nJ-hard:{hard_j[k]:.3f}",
                ha="center", fontsize=9)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("../img/anonymizer_intersection_easy_vs_hard.png", dpi=300)
    plt.show()


def main():
    files_easy = glob.glob("../experiment/speaker_linkability/easy_speakers/*.txt")
    files_hard = glob.glob("../experiment/speaker_linkability/hard_speakers/*.txt")

    print(f"{len(files_easy)} easy files found")
    print(f"{len(files_hard)} hard files found")

    groups_easy = group_files_by_arch_L(files_easy)
    groups_hard = group_files_by_arch_L(files_hard)

    compare_anonymizers(groups_easy, groups_hard)

if __name__ == "__main__":
    main()
