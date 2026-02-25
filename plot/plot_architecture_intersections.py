import glob
import matplotlib.pyplot as plt
import numpy as np
import os

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

def group_files_by_arch_anonymizer_L(files):
    groups = {}
    for f in files:
        base = os.path.basename(f).replace(".txt", "")
        parts = base.split("_")
        
        if "WavLM" in parts:
            arch = "WavLM_ECAPA" 
            anonymizer = parts[-2] 
            L = parts[-1]
        else:
            _, _, arch, anonymizer, L = parts

        spk = get_speakers_list(f)

        key = (anonymizer, L)
        if key not in groups:
            groups[key] = {}
        groups[key][arch] = spk
    return groups


def plot_results(results):
    """
    3 pairwise comparisons:
    - ECAPA vs ResNet
    - ECAPA vs WavLM_ECAPA
    - ResNet vs WavLM_ECAPA
    """
    labels = [f"{b}-{l}" for ((b, l), *_) in results]

    easy_inter_ER = [r[1] for r in results]   # ECAPA vs ResNet
    easy_union_ER = [r[2] for r in results]
    easy_j_ER = [r[3] for r in results]

    easy_inter_EW = [r[4] for r in results]   # ECAPA vs WavLM
    easy_union_EW = [r[5] for r in results]
    easy_j_EW = [r[6] for r in results]

    easy_inter_RW = [r[7] for r in results]   # ResNet vs WavLM
    easy_union_RW = [r[8] for r in results]
    easy_j_RW = [r[9] for r in results]

    hard_inter_ER = [r[10] for r in results]
    hard_union_ER = [r[11] for r in results]
    hard_j_ER = [r[12] for r in results]

    hard_inter_EW = [r[13] for r in results]
    hard_union_EW = [r[14] for r in results]
    hard_j_EW = [r[15] for r in results]

    hard_inter_RW = [r[16] for r in results]
    hard_union_RW = [r[17] for r in results]
    hard_j_RW = [r[18] for r in results]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    width = 0.18

    comparisons = [
        ("ECAPA vs ResNet", easy_inter_ER, easy_union_ER, easy_j_ER, 
         hard_inter_ER, hard_union_ER, hard_j_ER, 0),
        ("ECAPA vs WavLM_ECAPA", easy_inter_EW, easy_union_EW, easy_j_EW,
         hard_inter_EW, hard_union_EW, hard_j_EW, 1),
        ("ResNet vs WavLM_ECAPA", easy_inter_RW, easy_union_RW, easy_j_RW,
         hard_inter_RW, hard_union_RW, hard_j_RW, 2)
    ]

    for title, e_inter, e_union, e_j, h_inter, h_union, h_j, idx in comparisons:
        ax = axes[idx]
        
        offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
        
        ax.bar(x + offsets[0], e_inter, width, label="Easy Intersection", color="#ec6a37")
        ax.bar(x + offsets[1], e_union, width, label="Easy Union", color="#d30404d8")
        ax.bar(x + offsets[2], h_inter, width, label="Hard Intersection", color="#05b605")
        ax.bar(x + offsets[3], h_union, width, label="Hard Union", color="#127712E4")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Speaker number")
        ax.set_title(title, fontsize=14, fontweight='bold', pad=-20)

        y_max = max(max(e_union + [0]), max(h_union + [0])) * 1.25
        ax.set_ylim(0, y_max)

        for k in range(len(labels)):
            ymax = max(e_union[k] if k < len(e_union) else 0, 
                      h_union[k] if k < len(h_union) else 0)
            ax.text(k, ymax + 0.03*y_max,
                    f"J-easy:{e_j[k]:.3f}\nJ-hard:{h_j[k]:.3f}",
                    ha="center", fontsize=8)

        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Architecture Comparisons – Intersection / Union ", 
                 fontsize=16, fontweight='bold')
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 0.955), 
               ncol=4, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("../img/attacker_intersection_3architectures.png", dpi=300, bbox_inches='tight')
    plt.show()


def compare_architectures(groups_easy, groups_hard):
    all_keys = sorted(set(groups_easy.keys()) | set(groups_hard.keys()))
    
    anonymizers = sorted(set([b for (b, _) in all_keys]))

    results = []

    for key in all_keys:
        anonymizer, L = key

        metrics = {}
        
        if key in groups_easy:
            g = groups_easy[key]
            
            if "ECAPA" in g and "ResNet" in g:
                inter = len(g["ECAPA"] & g["ResNet"])
                union = len(g["ECAPA"] | g["ResNet"])
                metrics['easy_ER'] = (inter, union, inter/union if union else 0)
            else:
                metrics['easy_ER'] = (0, 0, 0)
            
            if "ECAPA" in g and "WavLM_ECAPA" in g:
                inter = len(g["ECAPA"] & g["WavLM_ECAPA"])
                union = len(g["ECAPA"] | g["WavLM_ECAPA"])
                metrics['easy_EW'] = (inter, union, inter/union if union else 0)
            else:
                metrics['easy_EW'] = (0, 0, 0)
            
            if "ResNet" in g and "WavLM_ECAPA" in g:
                inter = len(g["ResNet"] & g["WavLM_ECAPA"])
                union = len(g["ResNet"] | g["WavLM_ECAPA"])
                metrics['easy_RW'] = (inter, union, inter/union if union else 0)
            else:
                metrics['easy_RW'] = (0, 0, 0)
        else:
            metrics['easy_ER'] = (0, 0, 0)
            metrics['easy_EW'] = (0, 0, 0)
            metrics['easy_RW'] = (0, 0, 0)

        if key in groups_hard:
            g = groups_hard[key]
            
            if "ECAPA" in g and "ResNet" in g:
                inter = len(g["ECAPA"] & g["ResNet"])
                union = len(g["ECAPA"] | g["ResNet"])
                metrics['hard_ER'] = (inter, union, inter/union if union else 0)
            else:
                metrics['hard_ER'] = (0, 0, 0)
            
            if "ECAPA" in g and "WavLM_ECAPA" in g:
                inter = len(g["ECAPA"] & g["WavLM_ECAPA"])
                union = len(g["ECAPA"] | g["WavLM_ECAPA"])
                metrics['hard_EW'] = (inter, union, inter/union if union else 0)
            else:
                metrics['hard_EW'] = (0, 0, 0)
            
            if "ResNet" in g and "WavLM_ECAPA" in g:
                inter = len(g["ResNet"] & g["WavLM_ECAPA"])
                union = len(g["ResNet"] | g["WavLM_ECAPA"])
                metrics['hard_RW'] = (inter, union, inter/union if union else 0)
            else:
                metrics['hard_RW'] = (0, 0, 0)
        else:
            metrics['hard_ER'] = (0, 0, 0)
            metrics['hard_EW'] = (0, 0, 0)
            metrics['hard_RW'] = (0, 0, 0)

        results.append((
            (anonymizer, L),
            *metrics['easy_ER'], *metrics['easy_EW'], *metrics['easy_RW'],
            *metrics['hard_ER'], *metrics['hard_EW'], *metrics['hard_RW']
        ))

    for anonymizer in anonymizers:
        metrics = {}
        
        easy_sets = {'ECAPA': [], 'ResNet': [], 'WavLM_ECAPA': []}
        for (b, l), d in groups_easy.items():
            if b == anonymizer:
                for arch in easy_sets.keys():
                    if arch in d:
                        easy_sets[arch].append(d[arch])
        
        easy_unions = {}
        for arch, sets in easy_sets.items():
            if sets:
                easy_unions[arch] = set.union(*sets)
            else:
                easy_unions[arch] = set()
        
        if easy_unions['ECAPA'] or easy_unions['ResNet']:
            inter = len(easy_unions['ECAPA'] & easy_unions['ResNet'])
            union = len(easy_unions['ECAPA'] | easy_unions['ResNet'])
            metrics['easy_ER'] = (inter, union, inter/union if union else 0)
        else:
            metrics['easy_ER'] = (0, 0, 0)
        
        if easy_unions['ECAPA'] or easy_unions['WavLM_ECAPA']:
            inter = len(easy_unions['ECAPA'] & easy_unions['WavLM_ECAPA'])
            union = len(easy_unions['ECAPA'] | easy_unions['WavLM_ECAPA'])
            metrics['easy_EW'] = (inter, union, inter/union if union else 0)
        else:
            metrics['easy_EW'] = (0, 0, 0)
        
        if easy_unions['ResNet'] or easy_unions['WavLM_ECAPA']:
            inter = len(easy_unions['ResNet'] & easy_unions['WavLM_ECAPA'])
            union = len(easy_unions['ResNet'] | easy_unions['WavLM_ECAPA'])
            metrics['easy_RW'] = (inter, union, inter/union if union else 0)
        else:
            metrics['easy_RW'] = (0, 0, 0)

        hard_sets = {'ECAPA': [], 'ResNet': [], 'WavLM_ECAPA': []}
        for (b, l), d in groups_hard.items():
            if b == anonymizer:
                for arch in hard_sets.keys():
                    if arch in d:
                        hard_sets[arch].append(d[arch])
        
        hard_unions = {}
        for arch, sets in hard_sets.items():
            if sets:
                hard_unions[arch] = set.union(*sets)
            else:
                hard_unions[arch] = set()
        
        if hard_unions['ECAPA'] or hard_unions['ResNet']:
            inter = len(hard_unions['ECAPA'] & hard_unions['ResNet'])
            union = len(hard_unions['ECAPA'] | hard_unions['ResNet'])
            metrics['hard_ER'] = (inter, union, inter/union if union else 0)
        else:
            metrics['hard_ER'] = (0, 0, 0)
        
        if hard_unions['ECAPA'] or hard_unions['WavLM_ECAPA']:
            inter = len(hard_unions['ECAPA'] & hard_unions['WavLM_ECAPA'])
            union = len(hard_unions['ECAPA'] | hard_unions['WavLM_ECAPA'])
            metrics['hard_EW'] = (inter, union, inter/union if union else 0)
        else:
            metrics['hard_EW'] = (0, 0, 0)
        
        if hard_unions['ResNet'] or hard_unions['WavLM_ECAPA']:
            inter = len(hard_unions['ResNet'] & hard_unions['WavLM_ECAPA'])
            union = len(hard_unions['ResNet'] | hard_unions['WavLM_ECAPA'])
            metrics['hard_RW'] = (inter, union, inter/union if union else 0)
        else:
            metrics['hard_RW'] = (0, 0, 0)

        results.append((
            (anonymizer, "GLOBAL"),
            *metrics['easy_ER'], *metrics['easy_EW'], *metrics['easy_RW'],
            *metrics['hard_ER'], *metrics['hard_EW'], *metrics['hard_RW']
        ))

    plot_results(results)


def main():
    files_easy = glob.glob("../experiment/speaker_linkability/easy_speakers/*.txt")
    files_hard = glob.glob("../experiment/speaker_linkability/hard_speakers/*.txt")

    print(f'{len(files_easy)} easy files found')
    print(f'{len(files_hard)} hard files found')

    groups_easy = group_files_by_arch_anonymizer_L(files_easy)
    groups_hard = group_files_by_arch_anonymizer_L(files_hard)

    all_archs_easy = set()
    all_archs_hard = set()
    for d in groups_easy.values():
        all_archs_easy.update(d.keys())
    for d in groups_hard.values():
        all_archs_hard.update(d.keys())
    
    print(f'\nArchitectures found in easy: {sorted(all_archs_easy)}')
    print(f'Architectures found in hard: {sorted(all_archs_hard)}')

    compare_architectures(groups_easy, groups_hard)


if __name__ == "__main__":
    main()