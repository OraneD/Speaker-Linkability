This repository is an implementation of the Linkability metric for voice anonymization evaluation depicted in (article)[lien] and the reproduction of the experiment they led on the CommonVoice Dataset.

---
## Data
### Test (CommonVoice)
You can download the 11th version of CommonVoice english here : https://datacollective.mozillafoundation.org/organization/cmfh0j9o10006ns07jq45h7xk

Lists of the utterances and speaker ID for the A (enrollment) et B (trial) is available in the repo : [cv11-A](list-cv11-A) [TO DO](list-cv11-B)

### Train LibriSpeech-train-clean-360
The training set for the attackers is available here : https://www.openslr.org/12 

## Anonymization 
Training and test data should be anonymized with the same anonymizers and same parameters. The anonymizers used in this experiment are Baseline 3 and Baseline 5 of the VoicePrivacyChallenge 2025. 

We used the implementation of the authors of those anonymizers instead of the VoicePrivacyChallenge's implementation which is less practical, they are available here : [B3](https://github.com/DigitalPhonetics/speaker-anonymization) [B5](https://github.com/deep-privacy/SA-toolkit).

Note that this step can take a lot of time depending on your ressources : the train set is not such a challenge to anonymize due to its rather small size but anonymizing the CommonVoice dataset in a reasonable amount of time requires a lot of ressources, especially for B3. As an indication : 
* Anonymizing CommonVoice with B3 took around 20 hours with 70 nodes of 4 32Go GPUs
* Anonymizing CommonVoice with B5 took ~48 hours with 1 node of 4 32Go GPUs

Note that multi-node computation is not implemented for either anonymizer and that you would have to implement it yourself or launch several jobs in parallel to reduce computation time. 

## Attackers 

## Metric Description 
The Linkability metric has been created to evaluate the second criteria defining anonymized data : *data can be considered anonymous if it is not possible to link records that pertain to the same data subject*

In the case of speech anonymization, Linkability measures the risk that an attacker matches a speaker embedding $x_{test}$ computed from one or more anonymized test utterances, with the enrollment embedding $x_{enroll}$  corresponding to the same speaker $i$ among a set of $N'$ enrollment speakers.

Linkage succeeds if:  

$$
s(x_{\text{test}}, x_{\text{enroll}}^i) > \max_{j \neq i} s(x_{\text{test}}, x_{\text{enroll}}^j)
$$

The Linkability metric is defined as the probability of linkage over all test data:

$$
\pi_{\text{link}} = \Pr \left( s(x_{\text{test}}^i, x_{\text{enroll}}^i) > \max_{j \neq i} s(x_{\text{test}}^i, x_{\text{enroll}}^j) \right)
$$

A low $\pi_{\text{link}}$ indicates that it is difficult for an attacker to correctly  
link anonymized recordings to the correct speaker, thus supporting the claim of effective anonymization.

## Installation
TO DO

## Usage
The metric computation revolves around two classes `SimMatrix` & `Scores`

#### SimMatrix
The `SimMatrix` class aims to compute the Cosine Similarity Matrix pairing each test utterances against each enroll utterances. 
It takes as arguments :
- `enroll_path` and `trial_path` :  the pickle files (test and enroll set) containing a dictionary of speaker embeddings of shape `embedding_dic[speaker_id] = List[speaker_embeddings]`
- `L` :  which is the number of utterances used to compute the embedding of each test speaker
- `seed` : to initialize the random number generator involved in the choice of the utterances chosen to compute the average embeddings
- `matrix_path` : if the SimMatrix has already been computed and saved
It then computes and saves the matrix if not already done.

#### Scores
TO DO

### Result Example
![all_scores](img/linkability_all_scores.png)




