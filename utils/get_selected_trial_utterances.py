import pickle 
import random
from tqdm import tqdm


random.seed(42)
trial_embeddings = pickle.load(open("../data/spk2embs_with_id_cv11-B_Vox2_libri-54.pkl", "rb"))
print(type(trial_embeddings))

def get_trial_utterances(L):
    selected_utterances_dict = {}
    with open(f"../experiment/trial_utt_selected/trial_utt_selected_L-{L}.csv", "w") as file :
        for speaker in tqdm(sorted(trial_embeddings), total=len(trial_embeddings)):
            if L == 1 :
                selected_utterance = random.sample(trial_embeddings[speaker], 1)
                selected_utterances_dict[speaker] = selected_utterance[0]
                file.write(f"{speaker},{selected_utterance[0][0]}\n")
            if L > 1 : 
                if len(trial_embeddings[speaker]) >= L: 
                    selected_utterances = random.sample(trial_embeddings[speaker], L)
                    selected_utterances_dict[speaker] = [x[0] for x in selected_utterances]
                    file.write(f"{speaker},{','.join([x[0] for x in selected_utterances])}\n")
                if len(trial_embeddings[speaker]) == 1 :  
                    selected_utterances_dict[speaker] = trial_embeddings[speaker][0]
                if len(trial_embeddings[speaker]) < L and len(trial_embeddings[speaker]) != 1 :
                    selected_utterances_dict[speaker] = [x[0] for x in trial_embeddings[speaker]]
                    file.write(f"{speaker},{','.join([x[0] for x in selected_utterances])}\n")
    with open(f"../experiment/trial_utt_selected/trial_utt_selected_L-{L}.pkl", "wb") as handle :
        pickle.dump(selected_utterances_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
get_trial_utterances(1)
get_trial_utterances(3)
get_trial_utterances(30)

