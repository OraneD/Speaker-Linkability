from matrix import SimMatrix
import torchvision
import argparse
import yaml
torchvision.disable_beta_transforms_warning()


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type=str,
                        help="config file path")

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

    with open(args.config, 'r') as file :
        config = yaml.safe_load(file)

    N_ENROLL_SPK = config['matrix']['n_enroll_spk']
    N_enrolls = [n := N_ENROLL_SPK // (2 ** i) for i in range(0, 15) if N_ENROLL_SPK // (2 ** i) >= 20]

    if args.matrix_path != "None" :
          matrix = SimMatrix(config['matrix']['enroll_path'], 
                        config['matrix']['trial_path'], 
                        args.L, 
                        42, 
                        config['matrix']['model_type'], 
                        config['matrix']['data_type'],
                        args.matrix_path)
    else :
     matrix = SimMatrix(config['matrix']['enroll_path'], 
                        config['matrix']['trial_path'], 
                        args.L, 
                        42, 
                        config['matrix']['model_type'], 
                        config['matrix']['data_type'])

    for seed in config['matrix']['seeds'] : 
        for N in N_enrolls :
             matrix.get_scores_sequential(N,seed)

main()

