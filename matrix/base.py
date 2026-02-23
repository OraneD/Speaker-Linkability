
import torch
import torch.nn.functional as F
import pickle
import random
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from embedding import read_pkl, EmbeddingSet
from utils import get_avg_tensor, setup_logger
import sys

#TO DO : 
#        - Add DocString

class SimMatrix():
    def __init__(self, enroll_path, trial_path,L,seed, model_type, data_type, matrix_path=None):
        with open(enroll_path, "rb") as handle :
            self.enroll_embeddings = pickle.load(handle)
        with open(trial_path, "rb") as handle :
            self.trial_embeddings = pickle.load(handle)
        self.L = L
        self.seed = seed
        random.seed(seed)
        self.model_type = model_type
        self.data_type = data_type
        self.logger = setup_logger("logs", f"SimMatrix_L-{self.L}")
        with open("matrix/cv11-A_enroll_ids.pkl", "rb") as handle :
            self.enroll_ids = pickle.load(handle)
        with open("matrix/cv11-B_trial_ids.pkl", "rb") as handle :
            self.trial_ids = pickle.load(handle)     
        self.__verify_id_alignment()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpus = os.cpu_count()
        self.logger.info(f"Generating Cosine Similarity matrix for L = {self.L} and seed = {self.seed} ")
        self.logger.info(f"Device selected : {self.device}, CPUs available : {self.cpus}")
        with open("matrix/fixed_trial_selections.pkl", "rb") as handle :
            self.trial_selections = pickle.load(handle)
        self.trial_matrix = self.__get_trial_matrix()
        self.enroll_matrix = self.__get_enroll_matrix()
        self.similarity_matrix = self.__compute_cosine_similarity() if matrix_path == None else torch.load(matrix_path, map_location=torch.device(self.device))
        if matrix_path != None :
            self.logger.info(f"Cosine Similarity matrix loaded from file {matrix_path}")

    def __getitem__(self, index):
        if isinstance(index,int):
            return self.similarity_matrix[index][index]
        if isinstance(index,str):
            return self.enroll_ids[index]
        raise TypeError("Invalid argument type, must be int or str")
    
    def __verify_id_alignment(self):
        missing = [spk for spk in self.trial_ids if spk not in self.enroll_ids]
        if missing:
            self.logger.info(f"/!\ {len(missing)}  missing speaker in enroll")
            self.logger.info("Stopping matrix generation")
            sys.exit(1)
        else:
            self.logger.info("Every speaker of trial present in enroll")

        mismatched = [(spk, self.enroll_ids[spk], self.trial_ids[spk])
                    for spk in self.trial_ids
                    if spk in self.enroll_ids and self.enroll_ids[spk] != self.trial_ids[spk]]
        if mismatched:
            self.logger.info(f"/!\ {len(mismatched)} have mismatched ids")
            self.logger.info("Examples:", mismatched[:10])
            self.logger.info("Stopping matrix generation")
            sys.exit(1)
        else:
            self.logger.info("No mismatch between enroll and trial ids")
    

    def __get_enroll_matrix(self):
            self.logger.info("Generating enroll matrix...")
            embeddings_list = [self.enroll_embeddings[spk_id].clone().detach() for spk_id in sorted(self.enroll_ids, key=self.enroll_ids.get)]
            return torch.stack(embeddings_list) 

    def __get_trial_matrix(self):
            self.logger.info("Generating trial matrix from fixed selections...")
            embeddings_list = []

            trial_selection_L = self.trial_selections[self.L]

            for spk_id in sorted(self.trial_ids, key=self.trial_ids.get):
                if spk_id not in trial_selection_L:
                    self.logger.warning(f"No selection found for speaker {spk_id} at L={self.L}, skipping.")
                    continue

                selected_utt_ids = trial_selection_L[spk_id]

                speaker_utts = self.trial_embeddings[spk_id]
                utt_dict = {utt_id: emb for utt_id, emb in speaker_utts}

                selected_embs = []
                for utt_id in selected_utt_ids:
                    if utt_id in utt_dict:
                        selected_embs.append(utt_dict[utt_id])
                    else:
                        self.logger.warning(f"Utterance {utt_id} not found for speaker {spk_id} — skipping.")

                if not selected_embs:
                    self.logger.warning(f"No valid embeddings found for speaker {spk_id}, skipping.")
                    continue

                if len(selected_embs) > 1:
                    avg_emb = get_avg_tensor(selected_embs).clone().detach()
                else:
                    avg_emb = selected_embs[0].clone().detach()
                embeddings_list.append(avg_emb)

            self.logger.info(f"Trial matrix built for L={self.L} with {len(embeddings_list)} speakers.")
            return torch.stack(embeddings_list)

    def compute_random_avg(self, lst_tensor):
        if len(lst_tensor) >= self.L: # If there is more embeddings then needed for 1 speaker, we randomly select L embeddings
         selected_embeddings = random.sample(lst_tensor, self.L)
         return get_avg_tensor(selected_embeddings)
        if len(lst_tensor) == 1 :   # If there's only 1 embedding for the speaker, we return the one embbeding (it's the case for 1 speaker in whole dataset)
            return lst_tensor[0]                      # If there's not enough embedding for the speaker, we compute the average embedding with all embeddings from the speaker anyway
        return get_avg_tensor(lst_tensor) 

    def __compute_cosine_similarity(self):
        self.logger.info(f"Generating cosine similarity matrix of size ({len(self.trial_ids)},{len(self.enroll_ids)})")
        trial_matrix = F.normalize(self.trial_matrix).to(self.device)
        enroll_matrix = F.normalize(self.enroll_matrix).to(self.device)
        cosine_matrix = torch.mm(trial_matrix, enroll_matrix.T)
        torch.save(cosine_matrix, f"data/final_matrix/{self.model_type}_cosine_matrix_L-{self.L}_seed-{self.seed}_{self.data_type}.pt")
        self.logger.info(f"Done - file saved as /data/final_matrix/{self.model_type}_cosine_matrix_L-{self.L}_seed-{self.seed}_{self.data_type}.pt")
        return cosine_matrix
    

    def get_scores_sequential(self, N, seed):
        self.logger.info(f"Retrieving scores for {N} enroll speakers with seed {seed}")
        os.makedirs(f"experiment_2/{self.model_type}/matrix_L-{self.L}_{self.data_type}", exist_ok=True)
        torch.manual_seed(seed)
        device = "cpu"

        scores = {}

        sim_matrix = self.similarity_matrix.to(device)

        # /!\ need to change column variable into row it is in fact rows and not column
        for trial_idx in tqdm(list(self.trial_ids.values()), total=len(self.trial_ids.values()), desc="Computing scores..."):
            column = sim_matrix[trial_idx]  # We get the column for the corresponding speaker (every score of the current speaker vs all the enrolls)
            idx_column = torch.arange(len(column), device=column.device)
            mask = idx_column != trial_idx # We remove the score of the speaker vs himself so that it won't be randomly chosen
            idx_enroll = idx_column[mask]
            
            idx_sampled = idx_enroll[torch.randperm(len(idx_enroll))[:N-1]] # For each trial speaker, N-1 enroll speaker are randomly selected
            sampled_scores = column[idx_sampled]
            
            final_score = 1 if column[trial_idx] > max(sampled_scores) else 0  # Speaker has been successfully linked if his similarity with himself is the highest
            sampled_scores = torch.cat((sampled_scores, column[trial_idx].unsqueeze(0)))# We still add the score of the speaker against himself after de N-1 random draw
            all_idx = torch.cat((idx_sampled, torch.tensor([trial_idx], device=column.device)))
            

            # enroll_spk = [inversed_enroll_ids[idx.item()] for idx in all_idx]
            # scores[inversed_trial_ids[spk_id]] =  (final_score, enroll_spk)
            enroll_indices = [idx.item() for idx in all_idx]

            scores[trial_idx] = (final_score, enroll_indices)

        output_path = f"experiment_2/{self.model_type}/matrix_L-{self.L}_{self.data_type}/scores_N-{N}_seed-{seed}.pkl"
        with open(output_path, 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Done - file saved as {output_path}")




