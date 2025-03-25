import pickle
import glob
from tqdm import tqdm
from utils import setup_logger

class Scores():
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.seeds = [42, 0,  6, 7, 25]
        self.logger = setup_logger("logs", f"Scores-{'_'.join(self.dirpath).split('/')}")
        self.logger.info(f"Computing Linkability for Matrix {self.dirpath}")
        self.N_enrolls = [22024, 11012, 5506, 2753, 1376, 688, 344, 172, 86, 43, 21]
        self.all_scores = self.__get_all_scores()


    def __get_all_scores(self):
        all_results = {}
        for N in self.N_enrolls :
            self.logger.info(f"Computing overall linkability for {N} enroll spk ...")
            files = glob.glob(f"{self.dirpath}/*{N}*.pkl")
            for file in tqdm(files, total=len(files) , desc=f"Processing files for {N} enroll spk..."):
                with open(file, "rb") as handle :
                     score_dictionary = pickle.load(handle)
                     all_results[N] = self.compute_avg_score(score_dictionary)
        return all_results

    
    @staticmethod
    def compute_avg_score(score_dictionary):
        all_score = [x[0] for x in list(score_dictionary.values())]
        return sum(all_score)/len(all_score)

                     

        

