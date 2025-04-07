import torch 
from kiwano.embedding import read_pkl, write_pkl, EmbeddingSet
from tqdm import tqdm


"""
Takes a kaldi formatted list and an embeddings dictionary with utterance_id as key and embedding as value :
                    embeddings[utterance_id] = embedding
Saves a pickle dictionary with spk_id as key and averaged tensor of speaker (over all utterances of the speaker) as value :
                    averaged_embs[spk_id] = averaged_tensor

"""
KALDI_LST_PATH = "/lustre/fsn1/projects/rech/yjs/umf82pt/CommonVoice/cv11-A/liste" # Usefull just for set A anyway
EMBEDDINGS_PATH = "/lustre/fsn1/projects/rech/yjs/umf82pt/CommonVoice/embs/orig/embs_cv11-A_Vox2_libri-54.pkl"
OUTPUT_PATH = "../data/embs_avg_cv11-A_Vox2_libri-54.pkl"

def get_avg_tensor(lst_tensor):
    return torch.stack(lst_tensor).mean(dim=0)

def get_utt2spk(liste):
    return {utt.split()[0]: utt.split()[1] for utt in liste}

def get_spk2embs(embeddings,utt2spk):
    spk2embs = {}
    for utt_id, embs in tqdm(embeddings.h.items(), total=len(embeddings.h)):
        spk_id = utt2spk[utt_id]
        if spk_id not in spk2embs:
            spk2embs[spk_id] = []
            spk2embs[spk_id].append(embs)
        else : 
            spk2embs[spk_id].append(embs)
    return spk2embs

def get_averaged_embeddings(spk2embs):
    averaged_embeddings = EmbeddingSet()
    for spk_id, embs in tqdm(spk2embs.items(), total=len(spk2embs)):
        averaged_tensor = get_avg_tensor(embs)
        if averaged_tensor.numel() == 0 :
            raise ValueError(f"Empty tensor for speaker : {spk_id}")
        averaged_embeddings[spk_id] = averaged_tensor

    return averaged_embeddings


def main():
    liste = [x.strip() for x in open(KALDI_LST_PATH).readlines()]
    utt2spk = get_utt2spk(liste)
    embeddings = read_pkl(f"pkl:{EMBEDDINGS_PATH}")
    spk2embs = get_spk2embs(embeddings,utt2spk)
    averaged_embeddings = get_averaged_embeddings(spk2embs)
    print(f"Speakers : {len(spk2embs)}")
    print(f"Utterances : {sum([len(x) for x in list(spk2embs.values())])}")
    write_pkl(F"pkl:{OUTPUT_PATH}", averaged_embeddings)
    print(f"File saved as {OUTPUT_PATH}")
    out = read_pkl(f"pkl:{OUTPUT_PATH}")
    print(f"Speakers : {len(out.h)}")




if __name__=='__main__':
    main()