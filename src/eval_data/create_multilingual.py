import tqdm
import random
import argparse

from collections import Counter
from os.path import join
from src.utils.language_pairs import short2long, short2long_unseen


parser = argparse.ArgumentParser()
parser.add_argument("--collections_home", help="Path to original translation file (tsv) from mMARCO.", required=True)
parser.add_argument("--queries_home", help="Path to original translation files (tsv) from mMARCO.", required=True)
parser.add_argument("--include_unseen_languages", action='store_true')
args = parser.parse_args()

include_unseen_languages = args.include_unseen_languages

random.seed(0)
languages = list(short2long.values())
assert len(languages) == len(short2long)
languages.pop(languages.index("multilingual_6l"))
if include_unseen_languages:
    unseen_languages = list(short2long_unseen.values())
    unseen_languages.pop(unseen_languages.index("multilingual_14l"))
    languages += unseen_languages

# input: $DATA_HOME/google_translations/collections original (tsv) google translation files from mMARCO
collections_home = args.collections_home

# input: $DATA_HOME/google_translations/queries original (tsv) google query translation files from mMARCO
queries_home = args.queries_home

TSV_SEP = "\t"

def create_multilingual_corpus():
    print("Creating multilingual corpus file")
    lang2did_doc_tuples = {}
    for lang in languages:
        src_file = join(collections_home, lang + "_collection.tsv")
        print(f"Loading: {src_file}")
        with open(src_file, "r") as f:
            lang2did_doc_tuples[lang] = [line.strip().split(TSV_SEP) for line in f]

    tmp = []
    n_records = len(lang2did_doc_tuples[languages[0]])
    tgt_file = join(collections_home, f"multilingual_{len(languages)}l_collection.tsv")
    lang_mapping_file = tgt_file.replace(".tsv", ".did_lang-tuples.tsv")

    print(f"Writing {tgt_file} and {lang_mapping_file}")
    input("Continue?")
    with open(tgt_file, "w") as f, open(lang_mapping_file, "w") as g:
        for ith_record in tqdm.tqdm(range(n_records), total=n_records):
            # consistency check to ensure that loaded mmarco records are sorted by id
            assert all(lang2did_doc_tuples[lang][ith_record][0] == str(ith_record) for lang in languages)
            sampled_lang = languages[random.randint(0, len(languages) - 1)]
            tmp.append(sampled_lang)
            did, doc = lang2did_doc_tuples[sampled_lang][ith_record]
            f.write(did + TSV_SEP + doc + "\n")
            g.write(did + TSV_SEP + sampled_lang + "\n")

    print(f"language distribution: {Counter(tmp)}")


def create_multilingual_queryset():
    print("Creating multilingual query file")
    for mode in ["dev", "train"]:
        print(f"Running {mode}")
        lang2id2query = {}
        for lang in languages:
            id2query = {}
            if mode == "train":
                src_file = join(queries_home, mode, f"{lang}_queries.{mode}.tsv")
            else:
                src_file = join(queries_home, mode, f"{lang}_queries.{mode}.small.tsv")
            print(f"Loading {src_file}")
            with open(src_file, "r") as f:
                for line in f:
                    qid, query = line.strip().split(TSV_SEP)
                    id2query[qid] = query
            lang2id2query[lang] = id2query

        n_queries = len(lang2id2query[languages[0]])
        any_language = languages[0]
        if mode == "train":
            tgt_file = join(queries_home, mode, f"multilingual_{len(languages)}l_queries.{mode}.tsv")
        else:
            tgt_file = join(queries_home, mode, f"multilingual_{len(languages)}l_queries.{mode}.small.tsv")

        lang_mapping_file = tgt_file.replace(".tsv", ".qid_lang-tuples.tsv")
        tmp = []
        print(f"Writing {tgt_file} and {lang_mapping_file}")
        with open(tgt_file, "w") as f, open(lang_mapping_file, "w") as g:
            for qid in tqdm.tqdm(lang2id2query[any_language].keys(), total=n_queries):
                # consistency check
                assert all(qid in lang2id2query[l] for l in languages)
                sampled_language = languages[random.randint(0, len(languages)-1)]
                tmp.append(sampled_language)
                query = lang2id2query[sampled_language][qid]
                record = TSV_SEP.join([qid, query])
                f.write(record + "\n")
                g.write(sampled_language + "\n")

        print(f"language distribution: {Counter(tmp)}")


def main():
    create_multilingual_corpus()
    create_multilingual_queryset()


if __name__ == '__main__':
    main()
