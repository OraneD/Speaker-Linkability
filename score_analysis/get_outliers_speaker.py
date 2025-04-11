from scores import Scores
import pickle
import numpy as np



speaker_m1_resnet = pickle.load(open("experiment/scores/speaker_scores_matrix_L-1_Resnet_anon_B5.pkl", "rb"))
speaker_m3_resnet = pickle.load(open("experiment/scores/speaker_scores_matrix_L-3_Resnet_anon_B5.pkl", "rb"))
#speaker_m30_resnet = pickle.load(open("experiment/scores/speaker_scores_matrix_L-30_Resnet_anon_B5.pkl", "rb"))

speaker_m1_ecapa = pickle.load(open("experiment/scores/speaker_scores_matrix_L-1_ECAPA_anon_B5.pkl", "rb"))
speaker_m3_ecapa = pickle.load(open("experiment/scores/speaker_scores_matrix_L-3_ECAPA_anon_B5.pkl", "rb"))
#speaker_m30_ecapa = pickle.load(open("experiment/scores/speaker_scores_matrix_L-30_ECAPA_anon_B5.pkl", "rb"))

def get_quartile_repartition(scores):
    speaker_ids = list(scores.keys())
    values = list(scores.values())
    values_np = np.array(values)
    q1 = np.percentile(values_np, 25)
    q2 = np.median(values_np)
    q3 = np.percentile(values_np, 75)
    print(f"q1 = {q1}, q2 = {q2}, q3 = {q3}")

    speaker_q1 = [id_ for id_, val in scores.items() if val <= q1]
    speaker_q2 = [id_ for id_, val in scores.items() if q1 < val <= q2]
    speaker_q3 = [id_ for id_, val in scores.items() if q2 < val <= q3]
    speaker_q4 = [id_ for id_, val in scores.items() if val > q3]

    # print("Speakers in 1st quart  :", speaker_q1)
    # print("Speaker in 2nd quart :", speaker_q2)
    # print("Speaker in 3rd quart :", speaker_q3)
    print("Speaker in 4th quart :", len(speaker_q4))
    return speaker_q4

worst_spk_resnet_m1 = get_quartile_repartition(speaker_m1_resnet)
worst_spk_resnet_m3 = get_quartile_repartition(speaker_m3_resnet)
worst_spk_ecapa_m1 = get_quartile_repartition(speaker_m1_ecapa)
worst_spk_ecapa_m3 = get_quartile_repartition(speaker_m3_ecapa)

worst_spk_overall = []

for speaker in worst_spk_resnet_m1 : 
    if speaker in worst_spk_resnet_m3 and speaker in worst_spk_ecapa_m1 and speaker in worst_spk_ecapa_m3 :
        worst_spk_overall.append(speaker)

print(f"Worst speaker overall count : {len(worst_spk_overall)}")

with open("worst_spk_analysis/worst_spk_overall.txt", "w") as f :
    for speaker in worst_spk_overall :
        f.write(f"{speaker}\n")

