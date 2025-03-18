from matrix import SimMatrix
import torchvision
import argparse
torchvision.disable_beta_transforms_warning()


N_ENROLL_SPK = 20024

SEED_1 = 42
SEED_2 = 0
SEED_3 = 6
SEED_4 = 7
SEED_5 = 25

def main():
    matrix_1 = SimMatrix("data/embs_avg_cv11-A_Vox2_libri-54_anon_B5.pkl", "data/spk2embs_cv11-B_Vox2_libri-54_anon_B5.pkl",1, 42)
    matrix_3 = SimMatrix("data/embs_avg_cv11-A_Vox2_libri-54_anon_B5.pkl", "data/spk2embs_cv11-B_Vox2_libri-54_anon_B5.pkl",3, 42)
    matrix_30 = SimMatrix("data/embs_avg_cv11-A_Vox2_libri-54_anon_B5.pkl", "data/spk2embs_cv11-B_Vox2_libri-54_anon_B5.pkl",30, 42)
    matrices = [matrix_1, matrix_3, matrix_30]
    for matrix in matrices :
        for seed in [SEED_1, SEED_2, SEED_3, SEED_4, SEED_5] :
            for N in list(range(20, N_ENROLL_SPK, 100)) + [N_ENROLL_SPK]:
                matrix.get_scores(N, seed)

main()

