
from embedding import read_pkl
import torch
import torch.nn.functional as F
import pickle
import random
from tqdm import tqdm
import os
from utils import get_avg_tensor, setup_logger

#TO DO : - Save utterances picked during averaging in set B
#        - Implement Logger
#        - Implement __get_item() func
#        - Add DocString

class SimMatrix():
    def __init__(self, enroll_path, trial_path,L,seed, matrix_path=None):
        self.enroll_embeddings = read_pkl(f"pkl:{enroll_path}")
        with open(trial_path, "rb") as handle :
            self.trial_embeddings = pickle.load(handle)
        self.L = L
        self.seed = seed
        random.seed(seed)
        self.trial_ids = self.__map_trial_ids()
        self.enroll_ids = self.__map_enroll_ids()
        self.trial_matrix = self.__get_trial_matrix()
        self.enroll_matrix = self.__get_enroll_matrix()
        self.similarity_matrix = self.__compute_cosine_similarity() if matrix_path == None else torch.load(matrix_path)

    def __getitem__(self, index):
        if isinstance(index,int):
            return self.similarity_matrix[index][index]
        elif isinstance(index,str):
            return self.enroll_ids[index]
        else:
            raise TypeError("Invalid argument type, must be int or str")
    

    def __map_trial_ids(self):
        return {spk_id : idx for idx, spk_id in enumerate(list(self.trial_embeddings.keys()))}
    
    def __map_enroll_ids(self):
        enroll_ids = {spk_id: idx for idx, spk_id in enumerate(self.trial_embeddings.keys())}
        next_id = len(self.trial_embeddings)
        for spk_id in self.enroll_embeddings.h.keys():
            if spk_id not in self.trial_embeddings:
                enroll_ids[spk_id] = next_id
                next_id += 1
        return enroll_ids

    def __get_enroll_matrix(self):
            embeddings_list = [torch.tensor(self.enroll_embeddings.h[spk_id]) for spk_id in sorted(self.enroll_ids, key=self.enroll_ids.get)]
            return torch.stack(embeddings_list) 

    def __get_trial_matrix(self):
        if self.L == 1 :
            embeddings_list = [torch.tensor(random.sample(self.trial_embeddings[spk_id], 1)[0]) 
                                for spk_id in  sorted(self.trial_ids, key=self.trial_ids.get)]
            return torch.stack(embeddings_list)
        elif self.L > 1 :
            embeddings_list = [torch.tensor(self.compute_random_avg(self.trial_embeddings[spk_id])) 
                                for spk_id in  sorted(self.trial_ids, key=self.trial_ids.get)]
            return torch.stack(embeddings_list)

    def compute_random_avg(self, lst_tensor):
        if len(lst_tensor) >= self.L: # If there is more embeddings then needed for 1 speaker, we randomly select L embeddings
         selected_embeddings = random.sample(lst_tensor, self.L)
         return get_avg_tensor(selected_embeddings)
        elif len(lst_tensor) == 1 :   # If there's only 1 embedding for the speaker, we return the one embbeding (it's the case for 1 speaker in whole dataset)
            return lst_tensor[0]
        else :                        # If there's not enough embedding for the speaker, we compute the average embedding with all embeddings from the speaker anyway
         return get_avg_tensor(lst_tensor) 

    def __compute_cosine_similarity(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trial_matrix = F.normalize(self.trial_matrix).to(device)
        enroll_matrix = F.normalize(self.enroll_matrix).to(device)
        cosine_matrix = torch.mm(trial_matrix, enroll_matrix.T)
        torch.save(cosine_matrix, f"../data/final_matrix/cosine_matrix_L-{self.L}_seed-{self.seed}.pt")
        return cosine_matrix
    
    def get_scores(self, N, seed):
        os.makedirs(f"../experiment/matrix_L-{self.L}", exist_ok=True)
        torch.manual_seed(seed)
        transposed_sim_matrix = self.similarity_matrix.T
        inversed_enroll_ids = {v: k for k, v in self.enroll_ids.items()}
        inversed_trial_ids = {v: k for k, v in self.trial_ids.items()}
        scores_dictionary = {}
        for spk_id in tqdm(self.trial_ids.values(), total=len(self.trial_ids), desc="Computing score for each trial speaker..."):
            column = transposed_sim_matrix[spk_id] # We get the column for the corresponding speaker (every score of the current speaker vs all the enrolls)
            idx_column = torch.arange(len(column))
            mask = idx_column != spk_id # We remove the score of the speaker vs himself so that it won't be randomly chosen
            idx_enroll = idx_column[mask]
            idx_sampled = idx_enroll[torch.randperm(len(idx_enroll))[:N-1]] # For each trial speaker, N-1 enroll speaker are randomly selected
            sampled_scores = column[idx_sampled]
            final_score = 1 if column[spk_id] > max(sampled_scores) else 0 # Speaker has been successfully linked if his similarity with himself is the highest
            sampled_scores = torch.cat((sampled_scores, column[spk_id].unsqueeze(0)))# We still add the score of the speaker against himself after de N-1 random draw
            all_idx = torch.cat((idx_sampled, torch.tensor([spk_id])))
            enroll_spk = [inversed_enroll_ids[idx.item()] for idx in all_idx]
            scores_dictionary[inversed_trial_ids[spk_id]] = (final_score, enroll_spk)
            with open(f"../experiment/matrix_L-{self.L}/scores_N-{N}_seed-{seed}.pkl", 'wb') as handle:
                pickle.dump(scores_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)





                      
matrix = SimMatrix("../data/embs_avg_cv11-A_Vox2_libri-54_anon_B5.pkl", "../data/spk2embs_cv11-B_Vox2_libri-54_anon_B5.pkl",30, 42)
print(matrix.trial_matrix.shape)
print(matrix.similarity_matrix.shape)
matrix.get_scores(40, 42)

