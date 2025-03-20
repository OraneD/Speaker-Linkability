source /srv/storage/talc3@storage4.nancy.grid5000/multispeech/calcul/users/odufour/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/odufour/Anonymization_Linkability:$PYTHONPATH
echo "ok"
conda activate kiwano_env
echo "env ok"

L=$1
MATRIX_PATH=$2
N=$3
SEED=$4

python generate_matrices.py --L $L --matrix_path $MATRIX_PATH --N $N --seed $4
