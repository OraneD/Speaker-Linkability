import pickle as pkl
import glob
import re
import os
from tqdm import tqdm
from collections import defaultdict

def merge_scores(folder_path):

    """
    Takes matrix scores folder and return score per speaker : 
         {spk_id: {N: [score_seed1, score_seed2, ...]}}
    """

    folder = folder_path.split("/")[1]
    filename = folder_path.split("/")[-1]
    merged = defaultdict(lambda: defaultdict(list)) 
    files = sorted(glob.glob(f"{folder_path}/*.pkl"))
    pattern = re.compile(r"N-(\d+)_seed-(\d+)")  
    
    for f in files:
        m = pattern.search(f)
        if not m:
            print(f"Wrong input : {f}")
            continue
        N = int(m.group(1))
        seed = int(m.group(2))

        with open(f, "rb") as handle:
            data = pkl.load(handle)  
        for spk_id, score in data.items():
            merged[spk_id][N].append(score[0])

    merged = {spk: dict(scores) for spk, scores in merged.items()}
    os.makedirs(f"../experiment/speaker_linkability/{folder}", exist_ok=True)
    with open(f"../experiment/speaker_linkability/{folder}/{filename}.pkl", "wb") as f:
        pkl.dump(merged, f, protocol=pkl.HIGHEST_PROTOCOL)


def main():
    all_folders = glob.glob("experiment/*/*") # will take all the .pkl file of all matrices for all attacker/anonymizer pairs in the experiment folder
    for folder in tqdm(all_folders, total=len(all_folders), desc="Computing speaker scores..") :
        merge_scores(folder)




main()