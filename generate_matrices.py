from matrix import SimMatrix
import torchvision
import argparse
torchvision.disable_beta_transforms_warning()



N_ENROLL_SPK = 22024
SEEDS = [42, 0,  6, 7, 25]
ENROLL_PATH = "data/embs_avg_cv11-A_ECAPA.pkl"
TRIAL_PATH = "data/spk2embs_cv11-B_ECAPA.pkl"

N_enrolls = [n := N_ENROLL_SPK // (2 ** i) for i in range(0, 15) if N_ENROLL_SPK // (2 ** i) >= 20]

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--L",
                       type=int,
                       help="Number of trial utterances")
    
    parser.add_argument("--matrix_path",
                         default=None,
                         type=str,
                         help="path to the matrix to process if already existing")
    return parser
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.matrix_path != "None" :
     matrix = SimMatrix(ENROLL_PATH, TRIAL_PATH, args.L, 42, args.matrix_path)
    else :
     matrix = SimMatrix(ENROLL_PATH, TRIAL_PATH, args.L, 42)

    for seed in SEEDS : 
        for N in N_enrolls :
             matrix.get_scores_sequential(N,seed)

main()

