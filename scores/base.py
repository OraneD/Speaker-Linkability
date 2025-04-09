import pickle
import glob
import os
from tqdm import tqdm
from utils import setup_logger

"""
Takes a directory containing dictionaries matrix scores and computes :
        - The overall linkability across all seeds and N enrollment speaker for the given matrix
        - The overall linkability across all seeds and N enrollment speaker for all trial speakers of the given matrix
"""

class Scores():
    def __init__(self, dirpath, model):
        self.dirpath = dirpath
        self.model = model
        self.seeds = [42, 0,  6, 7, 25]
        self.logger = setup_logger("logs", f"Scores-{'_'.join(self.dirpath).split('/')}")
        self.logger.info(f"Computing Linkability for Matrix {self.dirpath}")
        self.N_enrolls = [22024, 11012, 5506, 2753, 1376, 688, 344, 172, 86, 43, 21]
        self.all_scores = self.__get_all_scores()
        self.speaker_scores = self.__get_speaker_scores()


    def __get_all_scores(self):
        all_results = {}
        for N in self.N_enrolls :
            self.logger.info(f"Computing overall linkability for {N} enroll spk ...")
            files = glob.glob(f"{self.dirpath}/*{N}*.pkl")
            for file in tqdm(files, total=len(files)):
                with open(file, "rb") as handle :
                     score_dictionary = pickle.load(handle)
                     all_results[N] = self.compute_avg_score(score_dictionary)
        return all_results
    
    def __get_speaker_scores(self):
        os.makedirs("experiment/scores", exist_ok=True)
        speaker_scores = {}
        files = glob.glob(f"{self.dirpath}/*.pkl")
        self.logger.info(f"Computing speaker scores for matrix {self.dirpath}...")
        for file in tqdm(files, total=len(files), desc=f"Retrieving scores for speakers..."):
            with open(file, "rb") as handle :
                score_dictionary = pickle.load(handle)
                for speaker,value in score_dictionary.items():
                    if speaker not in speaker_scores:
                        speaker_scores[speaker] = []
                    speaker_scores[speaker].append(value[0])
        self.logger.info(f"Speaker score computed for {len(speaker_scores)} speakers")
        avg_speaker_scores ={speaker:sum(value)/len(value) for speaker, value in speaker_scores.items()}
        with open(f"experiment/scores/speaker_scores_{self.dirpath.split('/')[-1]}_{self.model}.pkl", "wb") as handle :
            pickle.dump(avg_speaker_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return avg_speaker_scores


    @staticmethod
    def compute_avg_score(score_dictionary):
        all_score = [x[0] for x in list(score_dictionary.values())]
        return sum(all_score)/len(all_score)

                     

        

