worst_spk = [x.strip() for x in open("worst_spk_analysis/worst_spk_overall.txt").readlines()]

spk2utt_L1 = [x.strip() for x in open("../experiment/trial_utt_selected/trial_utt_selected_L-1.csv").readlines()]
spk2utt_L3 = [x.strip() for x in open("../experiment/trial_utt_selected/trial_utt_selected_L-3.csv").readlines()]

with open("worst_spk_analysis/worst_spk_utt.csv", "w") as file :
    for i in range(len(spk2utt_L1)) :
        if spk2utt_L1[i].split(",")[0] in worst_spk :
            file.write(f"{spk2utt_L1[i]},{' '.join(spk2utt_L3[i].split(',')[1:])}\n")