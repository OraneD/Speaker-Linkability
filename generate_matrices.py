from matrix import SimMatrix
import torchvision
import argparse
torchvision.disable_beta_transforms_warning()



N_ENROLL_SPK = 20024
SEEDS = [42, 0,  6, 7, 25]
ENROLL_PATH = "data/embs_avg_cv11-A_Vox2_libri-54_anon_B5.pkl"
TRIAL_PATH = "data/spk2embs_cv11-B_Vox2_libri-54_anon_B5.pkl"

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--L",
                       type=int,
                       help="Number of trial utterances")
    
    parser.add_argument("--matrix_path",
                         default=None,
                         type=str,
                         help="path to the matrix to process if already existing")
    
    parser.add_argument("--N",
                        default=20,
                        type=int,
                        help="First number of enroll speaker selected")
    
    parser.add_argument("--seed",
                        default=42,
                        type=int)
    return parser
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.matrix_path != None :
     matrix = SimMatrix(ENROLL_PATH, TRIAL_PATH, args.L, 42, args.matrix_path)
    else :
     matrix = SimMatrix(ENROLL_PATH, TRIAL_PATH, args.L, 42)

    for N in list(range(args.N,N_ENROLL_SPK , 100)) + [N_ENROLL_SPK] :
        matrix.get_scores_parallel(N,args.seed)

main()

