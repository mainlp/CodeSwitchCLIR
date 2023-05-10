"""
This script reads MS MARCO data (EN-EN)* and prepares code-switched training data EN_x-EN_y following BL-CS and ML-CS
appraoches explained in the paper.

*MS MARCO needs to be re-formatted into jsonl, run scripts/convert_collections.sh first (see README.md)
"""
import logging
import json
import os
import pickle
import tqdm
import math
import numpy as np
import random
import argparse

from copy import deepcopy
from os.path import join
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.corpus import stopwords
from typing import Dict, List
from src.utils.language_pairs import short2long
from src.utils.helper import chunk
from src.utils.language_pairs import mlingpairs_seen, langpairs_seen


# represents any language in multilingual code switching
TOKEN_ANY = "xx"

# flag is used to control whether to compute token ratio (as opposed to translation probability) over all words or
# only non-stopwords
count_all_words = False

# it's not necessary to load stopwords
en_stopwords = None if count_all_words else set(stopwords.words('english'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument("--datadir_enen", help="Folder containing EN-EN jsonl train/dev files (run train_data/baseline_finetuning.py first)", required=True)
parser.add_argument("--path_embeddings", help="Folder containing embedding files (run scripts/download.sh to download https://github.com/facebookresearch/MUSE#multilingual-word-embeddings).", required=True)
parser.add_argument("--output_dir", help="Output directory for writing code-switched training data.", required=True)
parser.add_argument("--datadir_enen", help="Input directory containing source file to be code-switched (train_csclir.jsonl).")
parser.add_argument("--path_embeddings", help="Input directory multilingual (MUSE) word embeddings.")
args = parser.parse_args()

# (required) first run: 'train_data/baseline_zeroshot.py'
# expects train_csclir.jsonl in datadir_enen (we use Reimer's script to create training data from MS MARCO).
# input: $DATA_HOME/finetuning/enen folder (requires to run train_data/baseline_finetuning.py first)
datadir_enen = args.datadir_enen

# input: $DATA_DIR/muse_embeddings folder (scripts/download.sh)
path_embeddings = args.path_embeddings

# output: $DATA_DIR/CLWE-CS folder
tgt_basedir = args.output_dir

# output: $DATA_DIR/.cache folder
cache_dir = os.path.join(os.path.abspath(os.path.join(tgt_basedir, os.pardir)), ".cache")



def load_embs(lang:str):
  """
  Loads MUSE word embeddings (https://github.com/facebookresearch/MUSE#multilingual-word-embeddings)
  :param lang: language identifier
  :return: list of vocabulary terms, numpy embedding matrix [n_terms x emb_size]
  """
  file = join(path_embeddings, f"wiki.multi.{lang}.vec")
  logger.info("loading %s" % file)
  with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()
  lines.pop(0) # remove meta data
  vocab = []
  emb_matrix = []
  for line in lines:
    term = line.split(" ")[0]
    vec = np.array([float(elem) for elem in line.split(" ")[1:]])
    if 'مالكوفيتش'== term: # skip corrupt line (emb.size < 300)
      continue
    assert vec.size == 300
    vocab.append(term)
    emb_matrix.append(vec)
  emb_matrix = np.array(emb_matrix)
  return vocab, emb_matrix


def code_switch(train_dev: str, qlang: str, dlang: str, prob: float) -> None:
  """
  Load EN-EN MS MARCO data, load one (BL-CS) or multiple (ML-CS) bilingual lexicons, iterate all training instances
  and apply code-switching, i.e. turns EN-EN into EN_x-En_y.

  :param train_dev: (deprecated) "train" or "dev"
  :param qlang: query language X
  :param dlang: document language Y
  :param prob: translation probability p
  :return: None
  """
  random.seed(1)
  tgt_dir = join(tgt_basedir, f"translation_prob={prob}/{qlang + dlang}")
  new_train_dev_file = join(tgt_dir, f"{train_dev}_csclir.jsonl")

  if os.path.exists(new_train_dev_file):
    logger.info(f"{new_train_dev_file} exists already, skipping.")
    return
  else:
    logger.info(f"Target file: {new_train_dev_file}")

  corpus_vocab, query_vocab, records = load_enen_traindev(train_dev)
  en_vocab, en_emb = load_embs("en")
  mapping = deepcopy(short2long)

  # xx is not an actual language but a meta token referring to "any" language
  del mapping["xx"]

  # Load bilingual lexicon(s) for query language(s)
  query_translation_table = None
  lang2query_ttables = None
  if qlang == TOKEN_ANY:
    # ML-CS: Load one bilingual lexicon EN->L2 for each seen query language L2 (includes query and doc languages)
    lang2query_ttables = {
      qlang: get_translation_table_from_knn(qlang, query_vocab, en_emb, en_vocab, train_dev, "query")
      for qlang in mapping.keys()
      if qlang != "en" # No need to code-switch EN to EN
    }
  else:
    # BL-CS: Load single bilingual lexicon EN->QL (don't load lexicon if query language is EN)
    query_translation_table = get_translation_table_from_knn(
      qlang, query_vocab, en_emb, en_vocab, train_dev, "query"
    ) if qlang != "en" else None

  # Loading bilingual lexicon(s) for document language(s)
  doc_translation_table = None
  lang2doc_ttables = None
  if dlang == TOKEN_ANY:
    # ML-CS
    lang2doc_ttables = {
      dlang: get_translation_table_from_knn(dlang, corpus_vocab, en_emb, en_vocab, train_dev, "corpus")
      for dlang in mapping.keys()
      if dlang != "en"
    }
  else:
    # BL-CS
    doc_translation_table = get_translation_table_from_knn(
      dlang, corpus_vocab, en_emb, en_vocab, train_dev, "corpus"
    ) if dlang != "en" else None

  # Due to OOV tokens the token ratio between EN and tgt language is not the same as translation probability
  # Keep track of code-switch (cs) counts, i.e. how often we actually code-switched vs. how often we didn't.
  query_cs_count, query_nocs_count = 0, 0
  doc_cs_count, doc_nocs_count = 0, 0

  logger.info("Writing results to %s" % new_train_dev_file)
  os.makedirs(tgt_dir, exist_ok=True)
  detokenizer = TreebankWordDetokenizer()
  with open(new_train_dev_file, "w") as f:
    for record in tqdm.tqdm(records):

      # Query-side Code-Switching
      # Pivot language is EN, only code switch if query language is different from EN
      if qlang != "en":
        # code switch multilingually if X-EN or X-X, otherwise code switch bilingually
        if qlang == TOKEN_ANY:
          cs_tokens, cs_count, nocs_count = code_switch_multilingually(prob, lang2query_ttables, record["qtokens"])
        else:
          cs_tokens, cs_count, nocs_count = code_switch_bilingually(prob, query_translation_table, record["qtokens"])
        query_cs_count += cs_count
        query_nocs_count += nocs_count
        q = detokenizer.detokenize(cs_tokens)
        record["query"] = q

      # Document-side Code-Switching (same procedure as queries)
      if dlang != "en":
        if dlang == TOKEN_ANY:
          cs_tokens, cs_count, nocs_count = code_switch_multilingually(prob, lang2doc_ttables, record["dtokens"])
        else:
          cs_tokens, cs_count, nocs_count = code_switch_bilingually(prob, doc_translation_table, record["dtokens"])
        doc_cs_count += cs_count
        doc_nocs_count += nocs_count
        d = detokenizer.detokenize(cs_tokens)
        record["passage"] = d

      del record["dtokens"]
      del record["qtokens"]

      line = json.dumps(record)
      f.write(line + "\n")

  logger.info(f"{qlang}-{dlang}:\tQuery cs (no cs): {query_cs_count} ({query_nocs_count})\t"
              f"Doc cs (no cs): {doc_cs_count} ({doc_nocs_count})")


def code_switch_multilingually(chance:float, lang2translation_table:Dict[str,Dict[str,str]], tokens:List[str]):
  """
  Multilingual Code-Switching (ML-CS):
  - For each token, with probability p=chance,
  -- (1) sample a language specific translation table and
  -- (2) translate the token.

  :param: chance: translation probability
  :param: lang2translation_table: one language table for each word (obtained via nearest cross-lingual neighbors in CLWE spaces)
  :param: tokens: list of EN tokens to be translated.
  :return: code switched sequence, number of code switches performed, number of no code switches performed
  """
  code_switched = []
  languages = list(lang2translation_table.keys())
  cs_count = 0
  no_cs_count = 0
  for tok in tokens:
    is_content_word = tok not in en_stopwords
    if random.random() < chance:
      random_language = languages[random.randint(0, len(languages)-1)]
      random_dict = lang2translation_table[random_language]
      if tok.lower() in random_dict:
        code_switched.append(random_dict[tok.lower()])
        cs_count += 1 if is_content_word or count_all_words else 0
      else:
        code_switched.append(tok)
        no_cs_count += 1 if is_content_word or count_all_words else 0
    else:
      code_switched.append(tok)
      no_cs_count += 1 if is_content_word or count_all_words else 0
  return code_switched, cs_count, no_cs_count


def code_switch_bilingually(chance:float, query_translation_table:Dict[str,str], tokens:List[str]):
  """
  Bilingual Code-Switching (BL-CS):
  - For each token, with probability=chance, translate it into its nearest cross-lingual neighbor (lexicon lookup).

  :param chance: translation probability
  :param query_translation_table: lexicons mapping EN->foreign language terms (precomputed nearest-neighbor search in CLWE spaces)
  :param tokens: list of EN tokens to be translated
  :return: code switched sequence, number of code switches performed, number of no code switches performed
  """
  code_switched = []
  cs_count = 0
  no_cs_count = 0
  for tok in tokens:
    is_content_word = tok not in en_stopwords
    if random.random() < chance and tok.lower() in query_translation_table:
      code_switched.append(query_translation_table[tok.lower()])
      cs_count += 1 if is_content_word or count_all_words else 0
    else:
      code_switched.append(tok)
      no_cs_count += 1 if is_content_word or count_all_words else 0
  return code_switched, cs_count, no_cs_count


def load_enen_traindev(train_dev:str):
  """
  Loads MS MARCO corpus
  :param train_dev: (deprecated) Specify to load train or (internal, separate held out) dev corpus.
  :return: set of corpus tokens, set of query tokens, list of jsonl records (qid, query, pid, passage, label)
  """
  cache_file = join(cache_dir, f"{train_dev}_cache.pkl")

  if not os.path.exists(cache_file):
    logger.info("Loading EN-EN for the first time")
    query_vocab = set()
    corpus_vocab = set()
    tokenizer = TreebankWordTokenizer()
    with open(join(datadir_enen, f"{train_dev}_csclir.jsonl"), "r", encoding="utf-8") as f:
      lines = f.readlines()
    records = []
    for line in tqdm.tqdm(lines):
      record = json.loads(line)
      qtokens = tokenizer.tokenize(record["query"])
      record["qtokens"] = qtokens
      query_vocab.update({qt.lower() for qt in qtokens})

      dtokens = tokenizer.tokenize(record["passage"])
      corpus_vocab.update({dt.lower() for dt in dtokens})
      record["dtokens"] = dtokens
      records.append(record)

    result = corpus_vocab, query_vocab, records
    logger.info("Saving to %s" % cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
      pickle.dump(result, f)

  else:
    logger.info("Re-loading cached EN-EN from %s" % cache_file)
    with open(cache_file, "rb") as f:
      corpus_vocab, query_vocab, records = pickle.load(f)

  return corpus_vocab, query_vocab, records


def get_translation_table_from_knn(
        language:str,
        query_or_corpus_vocab:List[str],
        en_emb:np.array,
        en_vocab:List[str],
        train_or_dev:str,
        query_or_corpus:str
):
  """
  1. Load (cross-lingual/multilingual) word embeddings of target language
  2. Compute overlap between EN query/corpus vocabulary and EN word embedding vocabulary.
  3. Find for each EN term it's nearest (cosine) cross-lingual neighbor (assumes vectors are length normalized).
  4. Build translation table EN->{query,corpus}-language
  :param language: target language
  :param query_or_corpus_vocab: vocabulary derived from set of EN queries/passages
  :param en_emb: EN word embeddings mapped into cross-lingual/multilingual space
  :param en_vocab: EN word embedding vocabulary
  :param train_or_dev: (deprecated, we don't use held out dev set for model selection but train for fixed number of steps)
  :param query_or_corpus: "query"/"corpus" to denote that the translation table is build for query-/corpus-vocab
  :return: translation dictionary from EN to target language
  """
  cache_file = join(cache_dir, f"translation_table_cache_{language}_{train_or_dev}_{query_or_corpus}.pkl")
  if os.path.exists(cache_file):
    logger.info("loading %s" % cache_file)
    with open(cache_file, "rb") as f:
      return pickle.load(f)

  else:
    vocab, emb = load_embs(language)
    contained_terms = list(set(en_vocab).intersection(query_or_corpus_vocab))
    translation_table = {}
    chunk_size = len(contained_terms) // 100
    n_iter = math.ceil(len(contained_terms) / chunk_size)
    logger.info(f"Computing translation table (size: {len(contained_terms)} word translation pairs) {cache_file}")
    for vocab_chunk in tqdm.tqdm(chunk(contained_terms, chunk_size), total=n_iter):
      embeddings = np.array([en_emb[en_vocab.index(term)] for term in vocab_chunk])
      matmul = np.matmul(embeddings, emb.T)
      # knn_vocab = np.argsort(-matmul)[:, 0] # slow
      knn_vocab = np.argmax(matmul, axis=-1)
      for x, y_id in zip(vocab_chunk, knn_vocab.tolist()):
        translation_table[x] = vocab[y_id]

    logger.info("Save %s" % cache_file)
    with open(cache_file, "wb") as f:
      pickle.dump(translation_table, f)
    return translation_table


def main():
  langpairs = langpairs_seen + mlingpairs_seen
  # there is nothing to code-switch when the target languages for queries and documents are both English
  langpairs.pop(("en", "en"))

  for prob in [0.25, 0.5, 0.75, 1.0]:
    logger.info(f"Translation probability: {prob}")
    for qlang, dlang in langpairs:
      for mode in ["train"]:
        logger.info(f"Code-switching {qlang}{dlang} ({mode})")
        code_switch(mode, qlang, dlang, prob=prob)


if __name__ == '__main__':
  main()
