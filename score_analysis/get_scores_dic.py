from scores import Scores
import pickle
import numpy as np

MATRIX_L1_PATH_resnet_anon = "../experiment/ResNet/matrix_L-1"
MATRIX_L3_PATH_resnet_anon = "../experiment/ResNet/matrix_L-3"
MATRIX_L30_PATH_resnet_anon = "../experiment/ResNet/matrix_L-30"

MATRIX_L1_PATH_ECAPA_anon = "../experiment/ECAPA/matrix_L-1_anon_B5"
MATRIX_L3_PATH_ECAPA_anon = "../experiment/ECAPA/matrix_L-3_anon_B5"
MATRIX_L30_PATH_ECAPA_anon = "../experiment/ECAPA/matrix_L-30_anon_B5"

m1_scores = Scores(MATRIX_L1_PATH_resnet_anon, "Resnet_anon_B5")
m3_scores = Scores(MATRIX_L3_PATH_resnet_anon, "Resnet_anon_B5")
m30_scores = Scores(MATRIX_L30_PATH_resnet_anon, "Resnet_anon_B5")

m1_ecapa_scores = Scores(MATRIX_L1_PATH_ECAPA_anon, "ECAPA_anon_B5")
m3__ecapa_scores = Scores(MATRIX_L3_PATH_ECAPA_anon, "ECAPA_anon_B5")
m30__ecapa_scores = Scores(MATRIX_L30_PATH_ECAPA_anon, "ECAPA_anon_B5")