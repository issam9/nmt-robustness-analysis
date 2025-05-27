# A Representation Level Analysis of NMT Model Robustness to Grammatical Errors
Code for paper: A Representation Level Analysis of NMT Model Robustness to Grammatical Errors

## Setup
Follow the following steps to setup a conda environement:
```bash
conda create --name nmt-analysis python=3.10
conda activate nmt-analysis
python -m pip install -r requirements.txt
conda install conda-forge::pattern
spacy download fr_core_news_sm
spacy download en_core_web_sm
```

## Model Checkpoints
Links to the base model checkpoints that we used for our experiments:
| Model            | Checkpoint URL                                                   |
|------------------|------------------------------------------------------------------|
| NLLB            | [https://huggingface.co/facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| MBART           | [https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) |
| M2M100          | [https://huggingface.co/facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) |
| OPUS-MT-En-Es   | [https://huggingface.co/Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) |
| OPUS-MT-Fr-Es   | [https://huggingface.co/Helsinki-NLP/opus-mt-fr-es](https://huggingface.co/Helsinki-NLP/opus-mt-fr-es) |
| OPUS-MT-En-De   | [https://huggingface.co/Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) |
| OPUS-MT-En-It   | [https://huggingface.co/Helsinki-NLP/opus-mt-en-it](https://huggingface.co/Helsinki-NLP/opus-mt-en-it) |
| OPUS-MT-En-Nl   | [https://huggingface.co/Helsinki-NLP/opus-mt-en-nl](https://huggingface.co/Helsinki-NLP/opus-mt-en-nl) |


## Data
Download and extract data
```bash
./download_and_extract.sh
```

## Synthetic Noise
Our code in this section is based on [this repo](https://bitbucket.org/antonis/nmt-grammar-noise/src/master/), especially the English part.

Download the m2 file from [here](https://www.comp.nus.edu.sg/~nlp/conll14st.html).

Introduce errors:
```bash
./introduce_errors.sh
```
Sample finetuning and probing subsets:
```bash
./sample_subsets.sh
```
## Finetuning
Run the following script to finetune models: 
```bash
./run_finetuning.sh
```

Run the following script to evaluate models: 
```bash
./evaluate_robustness.sh
```

## Encodings
Get layer wise encodings to use for Probing, Representation Distance and Robustness Heads experiments:
```bash
./encode.sh    # Original model encodings for noisy train, dev and test sentences and clean test sentences
./masked_head_encode.sh         # Run this to to get layer encodings after masking each head from previous layer
```

Get attention weights which we use compute attention to POS tags:
```bash
./get_attention_weights.sh      # Get the attention weights 
```

## Probing
The code in this section is based on [this repo](https://github.com/chrisdavis90/ged-syntax-probing/tree/main)

Train probes:
```bash
./train_probes.sh
```
Evaluate the probes:
```bash
./evaluate_probes.sh
```

## Results
The "results" folder includes experiment results. In "results/figures" we include the figures of each section of the paper, and in "results/finetuning" folder we include BLEU and chrF results on clean and noisy data.

## Notebooks
You can visit "notebooks" folder for post-processing and visualization of the results and representations.