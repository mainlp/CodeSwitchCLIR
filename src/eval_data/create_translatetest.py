import os
import pickle

import tqdm
import logging
import math
import numpy as np
import argparse

from os.path import join
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from src.utils.language_pairs import langpairs_seen, short2long
from src.train_data.codeswitch_clwe import load_embs
from src.utils.helper import chunk


parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", help="Directory where bilingual lexicons will be stored.")
parser.add_argument("--mmarco_dir", help="Directory of mMARCO (original .tsv) translations.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
SEP = "\t"


path_bilingual_lexicons_cache = args.cache_dir
path_mmarco_google = args.mmarco_dir
path_translate_test_corpus = path_mmarco_google + "translate_test_collections"
path_translate_test_query = path_mmarco_google + "translate_test_queries"


def load(filepath):
    id2txt = {}
    logging.info(f"Loading {filepath}")
    with open(filepath, "r") as f:
        for line in tqdm.tqdm(f, total=get_line_count(filepath)):
            _id, txt = line.rstrip().split(SEP)
            id2txt[_id] = txt
    return id2txt


def extract_vocab(id2text):
    vocab = set()
    tokenizer = TreebankWordTokenizer()
    _ids = list(id2text.keys())
    logging.info("Extracting vocabulary")
    for _id in tqdm.tqdm(_ids):
        text = id2text[_id]
        tokens = tokenizer.tokenize(text.lower())
        vocab.update(tokens)
        id2text[_id] = (text, tokens)

    return vocab, id2text


def term_by_term_clwe_translation(id2txt_tokens, translation_tbl):
    detokenizer = TreebankWordDetokenizer()
    return {
        _id: detokenizer.detokenize([translation_tbl.get(tok, tok) for tok in tokens])
        for _id, (text, tokens) in tqdm.tqdm(id2txt_tokens.items(), total=len(id2txt_tokens))
    }


def save(id2txt, filepath):
    logging.info(f"Writing file {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines([SEP.join([_id, txt]) + "\n" for _id, txt in id2txt.items()])


tgt_emb_vocab, tgt_emb_matrix = load_embs("en")
assert all(t == t.lower() for t in tgt_emb_vocab)


def compute_translation_tbl(query_corpus_vocab, src_emb_vocab, src_emb_table, lang):

    os.makedirs(path_bilingual_lexicons_cache, exist_ok=True)
    path_bilingual_lexicon = os.path.join(path_bilingual_lexicons_cache, f"{lang}2en_tbl.pkl")
    if os.path.exists(path_bilingual_lexicon):
        logging.info(f"Loading {path_bilingual_lexicon}")
        with open(path_bilingual_lexicon, "rb") as f:
            return pickle.load(f)

    contained_terms = list(set(src_emb_vocab).intersection(query_corpus_vocab))
    chunk_size = len(contained_terms) // 100
    n_iter = math.ceil(len(contained_terms) / chunk_size)
    logging.info(f"Computing translation table")
    tbl = {}
    for vocab_chunk in tqdm.tqdm(chunk(contained_terms, n=chunk_size), total=n_iter):
        chunk_embs = np.array([src_emb_table[src_emb_vocab.index(t)] for t in vocab_chunk])
        matmul = np.matmul(chunk_embs, tgt_emb_matrix.T)
        nearest_neighbors = np.argmax(matmul, axis=-1)
        for x, y_id in zip(vocab_chunk, nearest_neighbors.tolist()):
            tbl[x] = tgt_emb_vocab[y_id]

    logging.info(f"Writing {path_bilingual_lexicon}")
    with open(path_bilingual_lexicon, "wb") as f:
        pickle.dump(tbl, f)

    return tbl


def get_line_count(filename):
    logging.info("...counting number of lines")
    with open(filename, "r") as f:
        return sum(1 for _ in f)


def main():
    os.makedirs(path_translate_test_corpus, exist_ok=True)
    os.makedirs(path_translate_test_query, exist_ok=True)

    for qlang, dlang in langpairs_seen:
        logging.info(f"Runnung {qlang} -> {dlang}")
        for lang, inputfile, outputfile in [
            (qlang, (join(path_mmarco_google, "queries", "dev", f"{short2long[qlang]}_queries.dev.small.tsv")),
             join(path_translate_test_query, "dev", f"{short2long[qlang]}_queries.dev.small.tsv")),
            (dlang, (join(path_mmarco_google, "collections", f"{short2long[dlang]}_collection.tsv")),
             join(path_translate_test_corpus, f"{short2long[dlang]}_collection.tsv"))
        ]:
            logging.info(f"Current language: {dlang}")

            if lang == "en":
                logging.info(f"Skipping EN")
                continue

            if os.path.exists(outputfile):
               logging.info(f"Skipping because {outputfile} exists.")

            id2txt = load(inputfile)
            vocab, id2txt = extract_vocab(id2txt)
            emb_vocab, emb_matrix = load_embs(qlang)
            lang2en_translation_tbl = compute_translation_tbl(query_corpus_vocab=vocab, src_emb_table=emb_matrix,
                                                              src_emb_vocab=emb_vocab, lang=lang)
            id2txt = term_by_term_clwe_translation(id2txt, lang2en_translation_tbl)
            save(id2txt, outputfile)


if __name__ == '__main__':
    main()
