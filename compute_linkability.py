"""
Computes overall linkability and plots result for 3 matrices with L=1, L=3 & L=30

"""

from scores import Scores, plot_overall_linkability, plot_boxplot_linkability

MATRIX_L1_PATH = "experiment/matrix_L_orig-1"
MATRIX_L3_PATH = "experiment/matrix_L_orig-3"
MATRIX_L30_PATH = "experiment/matrix_L_orig-30"


m1_scores = Scores(MATRIX_L1_PATH)
m3_scores = Scores(MATRIX_L3_PATH)
m3O_scores = Scores(MATRIX_L30_PATH)

def main():
    plot_overall_linkability([m1_scores.all_scores,m3_scores.all_scores,m3O_scores.all_scores],"linkability_all_scores_orig.png")
    plot_boxplot_linkability([m1_scores.speaker_scores,m3_scores.speaker_scores,m3O_scores.speaker_scores],"linkability_all_speakers_orig.png")

main()