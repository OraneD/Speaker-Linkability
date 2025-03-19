#!/bin/bash
#SBATCH --job-name=MatrixGen     
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=40             
#SBATCH --qos=qos_gpu-t3             
#SBATCH --partition=gpu_p13             
#SBATCH -A yjs@v100                
#SBATCH -C v100-16g                      
#SBATCH --time=20:00:00                   
#SBATCH --output=MatrixGen_%j.out        
#SBATCH --error=MatrixGen_%j.err


L=$1
MATRIX_PATH=$2
N=$3
SEED=$4

source ~/.bashrc
conda activate kiwano_env
python generate_matrices.py --L $L --matrix_path $MATRIX_PATH --N $N --seed $4

#sbatch launch_matrix_gen.sh 1 data/final_matrix/cosine_matrix_L-1_seed-42.pt 2020 42
#sbatch launch_matrix_gen.sh 3 data/final_matrix/cosine_matrix_L-3_seed-42.pt 20 42
#sbatch launch_matrix_gen.sh 30 data/final_matrix/cosine_matrix_L-30_seed-42.pt 20 42


