
import torch
import torch.nn.functional as F
import pickle
import random
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from embedding import read_pkl, EmbeddingSet
from utils import get_avg_tensor, setup_logger

#TO DO : 
#        - Add DocString

class SimMatrix():
    def __init__(self, enroll_path, trial_path,L,seed, matrix_path=None):
        #self.enroll_embeddings = read_pkl(f"pkl:{enroll_path}")
        with open(enroll_path, "rb") as handle :
            self.enroll_embeddings = pickle.load(handle)
        with open(trial_path, "rb") as handle :
            self.trial_embeddings = pickle.load(handle)
        self.L = L
        self.seed = seed
        random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpus = os.cpu_count()
        self.logger = setup_logger("logs", f"SimMatrix_L-{self.L}")
        self.logger.info(f"Generating Cosine Similarity matrix for L = {self.L} and seed = {self.seed} ")
        self.logger.info(f"Device selected : {self.device}, CPUs available : {self.cpus}")
        self.trial_ids = self.__map_trial_ids()
        self.enroll_ids = self.__map_enroll_ids()
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
    

    def __map_trial_ids(self):
        self.logger.info("Mapping trial ids...")
        return {spk_id : idx for idx, spk_id in enumerate(list(self.trial_embeddings.keys()))}
    
    def __map_enroll_ids(self):
        self.logger.info("Mapping enroll ids...")
        enroll_ids = {spk_id: idx for idx, spk_id in enumerate(self.trial_embeddings.keys())}
        next_id = len(self.trial_embeddings)
        for spk_id in self.enroll_embeddings.keys():
            if spk_id not in self.trial_embeddings:
                enroll_ids[spk_id] = next_id
                next_id += 1
        return enroll_ids

    def __get_enroll_matrix(self):
            self.logger.info("Generating enroll matrix...")
            embeddings_list = [self.enroll_embeddings[spk_id].clone().detach() for spk_id in sorted(self.enroll_ids, key=self.enroll_ids.get)]
            return torch.stack(embeddings_list) 

    def __get_trial_matrix(self):
        self.logger.info("Generating trial matrix...")
        if self.L == 1 :
            embeddings_list = [random.sample(self.trial_embeddings[spk_id], 1)[0].clone().detach() 
                                for spk_id in  sorted(self.trial_ids, key=self.trial_ids.get)]
            return torch.stack(embeddings_list)
        elif self.L > 1 :
            embeddings_list = [self.compute_random_avg(self.trial_embeddings[spk_id]).clone().detach() 
                                for spk_id in  sorted(self.trial_ids, key=self.trial_ids.get)]
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
        torch.save(cosine_matrix, f"data/final_matrix/ECAPA_cosine_matrix_L-{self.L}_seed-{self.seed}_orig.pt")
        self.logger.info(f"Done - file saved as /data/final_matrix/ECAPA_cosine_matrix_L-{self.L}_seed-{self.seed}_orig.pt")
        return cosine_matrix
    
    @staticmethod
    def compute_score(args):
        spk_id, transposed_sim_matrix, inversed_enroll_ids, inversed_trial_ids, N = args

        column = transposed_sim_matrix[spk_id]  # We get the column for the corresponding speaker (every score of the current speaker vs all the enrolls)
        idx_column = torch.arange(len(column), device=column.device)
        mask = idx_column != spk_id # We remove the score of the speaker vs himself so that it won't be randomly chosen
        idx_enroll = idx_column[mask]
        
        idx_sampled = idx_enroll[torch.randperm(len(idx_enroll))[:N-1]] # For each trial speaker, N-1 enroll speaker are randomly selected
        sampled_scores = column[idx_sampled]
        
        final_score = 1 if column[spk_id] > max(sampled_scores) else 0  # Speaker has been successfully linked if his similarity with himself is the highest
        sampled_scores = torch.cat((sampled_scores, column[spk_id].unsqueeze(0)))# We still add the score of the speaker against himself after de N-1 random draw
        all_idx = torch.cat((idx_sampled, torch.tensor([spk_id], device=column.device)))
        
        enroll_spk = [inversed_enroll_ids[idx.item()] for idx in all_idx]
        return inversed_trial_ids[spk_id], (final_score, enroll_spk)

    def get_scores_parallel(self, N, seed):
        self.logger.info(f"Retrieving scores for {N} enroll speakers with seed {seed}")
        os.makedirs(f"experiment/ECAPA/matrix_L-{self.L}_orig", exist_ok=True)
        torch.manual_seed(seed)
        device = "cpu"

        transposed_sim_matrix = self.similarity_matrix.T.to(device)
        inversed_enroll_ids = {v: k for k, v in self.enroll_ids.items()}
        inversed_trial_ids = {v: k for k, v in self.trial_ids.items()}

        args_list = [
            (spk_id, transposed_sim_matrix, inversed_enroll_ids, inversed_trial_ids, N)
            for spk_id in self.trial_ids.values()
        ]

        with ProcessPoolExecutor(max_workers=40) as executor: 
            results = list(tqdm(executor.map(self.compute_score, args_list), total=len(args_list), desc="Computing scores..."))

        scores_dictionary = dict(results)

        output_path = f"experiment/ECAPA/matrix_L-{self.L}_orig/scores_N-{N}_seed-{seed}.pkl"
        with open(output_path, 'wb') as handle:
            pickle.dump(scores_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Done - file saved as {output_path}")

    def get_scores_sequential(self, N, seed):
        self.logger.info(f"Retrieving scores for {N} enroll speakers with seed {seed}")
        os.makedirs(f"experiment/ECAPA/matrix_L-{self.L}_orig", exist_ok=True)
        torch.manual_seed(seed)
        device = "cpu"

        scores = {}

        transposed_sim_matrix = self.similarity_matrix.T.to(device)
        inversed_enroll_ids = {v: k for k, v in self.enroll_ids.items()}
        inversed_trial_ids = {v: k for k, v in self.trial_ids.items()}

        for spk_id in tqdm(list(self.trial_ids.values()), total=len(self.trial_ids.values()), desc="Computing scores..."):
            column = transposed_sim_matrix[spk_id]  # We get the column for the corresponding speaker (every score of the current speaker vs all the enrolls)
            idx_column = torch.arange(len(column), device=column.device)
            mask = idx_column != spk_id # We remove the score of the speaker vs himself so that it won't be randomly chosen
            idx_enroll = idx_column[mask]
            
            idx_sampled = idx_enroll[torch.randperm(len(idx_enroll))[:N-1]] # For each trial speaker, N-1 enroll speaker are randomly selected
            sampled_scores = column[idx_sampled]
            
            final_score = 1 if column[spk_id] > max(sampled_scores) else 0  # Speaker has been successfully linked if his similarity with himself is the highest
            sampled_scores = torch.cat((sampled_scores, column[spk_id].unsqueeze(0)))# We still add the score of the speaker against himself after de N-1 random draw
            all_idx = torch.cat((idx_sampled, torch.tensor([spk_id], device=column.device)))
            
            enroll_spk = [inversed_enroll_ids[idx.item()] for idx in all_idx]
            scores[inversed_trial_ids[spk_id]] =  (final_score, enroll_spk)

        output_path = f"experiment/ECAPA/matrix_L-{self.L}_orig/scores_N-{N}_seed-{seed}.pkl"
        with open(output_path, 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Done - file saved as {output_path}")




