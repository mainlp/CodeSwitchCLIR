# Boosting Zero-shot Cross-lingual Retrieval by Training on Artificially Code-Switched Data 
In this work we systematically investigate training cross-encoders on artificially code-switched data and the effect on monolingual retrieval (MoIR), cross-lingual retrieval (CLIR) and multilingual retrieval (MLIR). Preprint: [arxiv.org/abs/2305.05295](https://arxiv.org/abs/2305.05295)     

## Installation
Our code has been tested with Python 3.8, we recommend to set up a new conda environment:
```shell
conda create --name csclir --file requirements.txt
conda activate csclir
```
To be able to create jsonl records from MS MARCO (`scripts/convert_collections.sh`) you need to additionally an environment with [pygaggle](https://github.com/castorini/pygaggle#installation) installed. 

## Training Data
For brevity we use `$DATA_HOME` refer to the folder containing all mMARCO and code-switch data. We use `$MODEL_HOME` to refer to the folder containing all trained models. Alternatively to creating the datasets from scratch (steps below), you can also directly download code-switched MS MARCO training data [on huggingface](https://huggingface.co/datasets/rlitschk/csclir/tree/main) 🤗. 

### 1. Prepare mMARCO
To create artificially code-switched training data and multilingual evaluation data we use the [mMARCO dataset (Bonifacio et al. 2021)](https://github.com/unicamp-dl/mMARCO). This step is required for all data preparation steps listed below.  
- Copy `scripts/download.sh` into `$DATA_HOME`. This downloads all queries/documents for all 14 languages and multilingual word embeddings from [MUSE (Lample et al.2017)](https://github.com/facebookresearch/MUSE#multilingual-word-embeddings), which are required for Code-Switching and Translate-Test (Download size: 57 GB). 
- Copy `scripts/convert_collections.sh` into `$DATA_HOME` and run to create jsonl recrods (this step is adopted from [here](https://github.com/unicamp-dl/mMARCO#data-prep) and requires [pygaggle](https://github.com/castorini/pygaggle)).

### 2. Prepare Baseline Data
- **Zero-shot**: To create EN–EN training data (jsonl), run
```shell
python train_data/baseline_zeroshot.py --output_dir $DATA_HOME/zeroshot
```
- **Fine-tuning**: To create datasets for MoIR fine-tuning (DE–DE, IT–IT, etc.) and CLIR fine-tuning (EN–DE, EN–IT, etc.), run
```shell
python train_data/baseline_finetuning.py --output_dir $DATA_HOME/finetuning
````

### 3. Prepare Code-Switching Data
- **Bilingual Code-Switching (BL-CS) and Multilingual Code-Switching (ML-CS)**: Create code-switched training data for all language pairs (seen languages). This requires cross-lingual word embeddings (clwe) for all seen languages.
```shell
python train_data/codeswitch_clwe.py --datadir_enen $DATA_HOME/finetuning/enen --path_embeddings $DATA_HOME/muse_embeddings --output_dir $DATA_HOME/clwe-cs
```
- **Wiki-CS**: Run `train_data/codeswitch_wiki_1.ipynb` to prepare bilingual dictionaries from parallel Wikipedia page titles and to create bilingually WIKI-CS'ed data. To create one multilingual WIKI-CS data (used in paper), run 
```shell
python train_data/codeswitch_wiki_2.py
```


## Test Data
In addition to evaluating on English MS MARCO data (we use **dev.small.qrels**) we additionally construct the following evaluation sets.

### a) Create Translate-Test Data
- We follow [Roy et al. (2020)](https://aclanthology.org/2020.emnlp-main.477/) and use bilingual dictionaries to translate everything (term by term) into English.
```shell
python eval_data/create_translatetest.py --mmarco_dir $DATA_HOME/google_translations --cache_dir $DATA_HOME/.cache 
```

### b) Create Multilingual Retrieval (MLIR) Data
- From parallel queries/collections in mMARCO we create multilingual evaluation sets by sampling for each query- or doc-id the content from a randomly sampled language.  
- We evaluate on a multilingual corpus (EN–X), multilingual query set (X–EN) and both sides from different languages (X–X). 
- The following command creates multilingual corpus file and multilingual query file. 
  - Output file `multilingual_6l_collection.tsv`: Multilingual corpus for six seen languages, X<sub>seen</sub>.
  - (Optional) flag `--include_unseen_languages`: include all fourteen languages, X<sub>all</sub>. 
```shell
python eval_data/create_multilingual.py --collections_home $DATA_HOME/google_translations/collections --queries_home $DATA_HOME/google_translations/queries 
```


## Training and Evaluation 
- We use `train_ranker.py` to train all models, you can find our hyperparameters in `scripts/run.sh`. 
- Run `eval_ranker.py` to create and evaluates run files. This script writes run files (.pkl). 
- We compare different approaches, i.e. models trained on different datasets, with paired t-tests and Bonferroni correction (`eval_ttest.py`, requires run files).  

## Cite
If you use this repository, please consider citing our paper:
```bibtex
@misc{litschko2023boosting,
      title={Boosting Zero-shot Cross-lingual Retrieval by Training on Artificially Code-Switched Data}, 
      author={Robert Litschko and Ekaterina Artemova and Barbara Plank},
      year={2023},
      eprint={2305.05295},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
