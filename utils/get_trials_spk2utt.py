from utils import  get_utt2spk, get_spk2embs
from embedding import read_pkl
import pickle

"""
Saves a dictionary with spk_id : List(embeddings) from set B
"""
KALDI_LST_PATH = "/lustre/fsn1/projects/rech/yjs/umf82pt/CommonVoice/cv11-B/liste" # Usefull just for set B anyway
EMBEDDINGS_PATH = "/lustre/fsn1/projects/rech/yjs/umf82pt/CommonVoice/embs/orig/embs_cv11-B_Vox2_libri-54.pkl"
OUTPUT_PATH = "../data/spk2embs_with_id_cv11-B_Vox2_libri-54.pkl"

liste = [x.strip() for x in open(KALDI_LST_PATH).readlines()]
utt2spk = get_utt2spk(liste)
embeddings = read_pkl(f"pkl:{EMBEDDINGS_PATH}")
print(embeddings)
spk2embs = get_spk2embs(embeddings,utt2spk)
print(len(spk2embs))
with open(OUTPUT_PATH, 'wb') as handle:
    pickle.dump(spk2embs, handle, protocol=pickle.HIGHEST_PROTOCOL)