#!/bin/bash
#SBATCH --job-name=MatrixGen         
#SBATCH --partition=prepost                
#SBATCH --time=04:00:00                   
#SBATCH --output=MatrixGen_%j.out        
#SBATCH --error=MatrixGen_%j.err

config=$1
L=$2
MATRIX_PATH=${3:-None}

source ~/.bashrc
conda activate kiwano_env
python generate_matrices.py --config $config --L $L --matrix_path $MATRIX_PATH 
#sbatch launch_matrix_gen.sh 1 data/final_matrix/cosine_matrix_L-1_seed-42.pt
#sbatch launch_matrix_gen.sh 3 data/final_matrix/cosine_matrix_L-3_seed-42.pt
#sbatch launch_matrix_gen.sh 30 data/final_matrix/cosine_matrix_L-30_seed-42.pt


