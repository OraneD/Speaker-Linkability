from scores import Scores, plot_overall_linkability


m1_scores = Scores("experiment/matrix_L-1")
m3_scores = Scores("experiment/matrix_L-3")
m3O_scores = Scores("experiment/matrix_L-30")

plot_overall_linkability([m1_scores.all_scores,m3_scores.all_scores,m3O_scores.all_scores],"linkability_all_scores.png")

