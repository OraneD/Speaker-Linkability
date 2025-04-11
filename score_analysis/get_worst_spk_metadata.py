from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
VALIDATED_PATH = "/lustre/fsn1/projects/rech/yjs/umf82pt/CommonVoice/cv-corpus-11.0-2022-09-21/en/validated.tsv"

def get_validated_dic(validated_path):
    validated_dict = {}
    with open(validated_path, "r", encoding="utf-8") as f:
        for line in f:
            columns = line.strip().split("\t")
            loc_id = columns[0]
            sentence = columns[2]
            age = columns[5]
            gender = columns[6]
            accent = re.sub(r'\s*\(.*?\)', '',columns[7])
            if loc_id not in validated_dict :
                validated_dict[loc_id] = (sentence, age, gender, accent)
    return validated_dict

def get_worst_spk_dic(validated_dic):
    worst_spk = [x.strip() for x in open("worst_spk_analysis/worst_spk_overall.txt").readlines()]
    worst_spk_dict = {}
    for spk in worst_spk :
        worst_spk_dict[spk] = validated_dic[spk]
    with open('worst_spk_analysis/worst_spk_data.pkl', 'wb') as f :
        pickle.dump(worst_spk_dict, f)
    return worst_spk_dict

def get_all_spk_dic(validated_dict):
    worst_spk = [x.strip() for x in open("worst_spk_analysis/worst_spk_overall.txt").readlines()]
    new_dic = validated_dict.copy()
    for speaker in validated_dict :
        if speaker in worst_spk :
            del new_dic[speaker]
    return new_dic

def plot_attributes(worst_spk_dic, other_speakers):
    def extract_data(dic, label):
        data = []
        for v in dic.values():
            age = v[1] if v[1] != "" else "NA"
            gender = v[2] if v[2] != "" else "NA"
            accent = v[3] if v[3] != "" else "NA"
            data.append((age, gender, accent, label))
        return data

    data = extract_data(worst_spk_dic, "worst_speakers") + extract_data(other_speakers, "all_speakers")

    df = pd.DataFrame(data, columns=["age", "gender", "accent", "group"])

    accent_map = {
        "Australian English": "AUS",
        "England English": "ENG",
        "Canadian English": "CAN",
        "United States English": "US",
        "India and South Asia": "India",
        "Hong Kong English": "HK",
        "United States English,England English": "US + ENG",
        "New Zealand English": "NZ",
        "Irish English": "Ireland",
        "West Indies and Bermuda": "Caribbean",
        "NA": "NA"
    }

    df["accent"] = df["accent"].apply(lambda a: accent_map[a] if a in accent_map else "other")

    attributes = ["age", "gender", "accent"]
    colors = {'worst_speakers': 'orange', 'all_speakers': 'skyblue'}
    df = df[df["age"] != "age"]
    df = df[df["gender"] != "gender"]
    df = df[df["accent"] != "accent"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, attr in zip(axes, attributes):
        counts = df.groupby([attr, "group"]).size().reset_index(name='count')
        total_per_group = df.groupby("group").size().to_dict()
        counts['percent'] = counts.apply(lambda x: 100 * x['count'] / total_per_group[x['group']], axis=1)

        pivot_df = counts.pivot(index=attr, columns="group", values="percent").fillna(0)

        pivot_df.plot(kind="bar", ax=ax, color=[colors.get(g, 'grey') for g in pivot_df.columns], width=0.8)
        ax.get_legend().set_title(None)
        ax.set_title(attr)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel(attr)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("../img/worst_spk_attributes_comparison.png")


def main():
    validated_dict = get_validated_dic(VALIDATED_PATH)
    worst_spk_dict = get_worst_spk_dic(validated_dict)
    other_speakers = get_all_spk_dic(validated_dict)
    plot_attributes(worst_spk_dict, other_speakers)
main()

