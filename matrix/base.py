
from embedding import read_pkl
import torch
import torch.nn.functional as F
import pickle
import random
from utils import get_avg_tensor, setup_logger

#TO DO : - Save utterances picked during averaging in set B
#        - Implement Logger
#        - Implement __get_item() func
#        - Add DocString

class SimMatrix():
    def __init__(self, enroll_path, trial_path,L,seed):
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
        self.similarity_matrix = self.__compute_cosine_similarity()
    
    

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

                      
matrix = SimMatrix("../data/embs_avg_cv11-A_Vox2_libri-54_anon_B5.pkl", "../data/spk2embs_cv11-B_Vox2_libri-54_anon_B5.pkl",1, 42)
print(matrix.trial_matrix.shape)
print(matrix.similarity_matrix.shape)
print(matrix.similarity_matrix[:5, :5])

