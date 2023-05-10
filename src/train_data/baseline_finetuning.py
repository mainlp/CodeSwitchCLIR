import os
import json
import tqdm
import random

from typing import Dict
from collections import defaultdict
from src.utils.language_pairs import langpairs_seen, short2long
from src.utils.helper import load_instances


random.seed(1)
include_unseen = True
if include_unseen:
    from src.utils.language_pairs import short2long_unseen, mlingpairs_unseen
    short2long = {**short2long, **short2long_unseen}
    langpairs = mlingpairs_unseen


# input: $DATA_HOME/zeroshot folder  (run train_data/baseline_zeroshot.py first)
# zeroshot_dir = "/mounts/data/proj/rlitschk/data/ms-marco-reimers"
zeroshot_dir = "/mounts/data/proj/rlitschk/data/csclir_release/zeroshot"

# input: $DATA_HOME/google_translations folder (run scripts/download.sh and scripts/convert_collections.sh first)
# path_mmarco_googletranslations = "/mounts/data/proj/rlitschk/data/mmarco/google"
path_mmarco_googletranslations = "/mounts/data/proj/rlitschk/data/csclir_release/google_translations"

# output: $DATA_HOME/finetuning folder
# output_dir = "/mounts/data/proj/rlitschk/data/ms-marco-reimers/mmarco_google"
output_dir = "/mounts/data/proj/rlitschk/data/csclir_release/finetuning"


def load_collections()->Dict[str,Dict[str,str]]:
    """
    Load for each language the document collection
    :return: language-to-collection mapping (collection: docid2doc dictionary)
    """
    print("loading documents")
    lang2did2doc = defaultdict(dict)
    base_dir = os.path.join(path_mmarco_googletranslations, "collections_jsonl")
    doc_languages =  {v for _, v in langpairs_seen}
    for lang in doc_languages:
        print(lang)
        tgt_dir = os.path.join(base_dir, short2long[lang])
        for file in os.listdir(tgt_dir):
            instances = load_instances(filepath=os.path.join(base_dir, short2long[lang], file), total=1_000_000)
            assert all(instance["id"] not in lang2did2doc[lang] for instance in instances)
            for instance in instances:
                lang2did2doc[lang][instance["id"]] = instance["contents"].strip()
    return lang2did2doc


def load_queries()->Dict[str,Dict[str,str]]:
    """
    Load for each language all queries
    :return: language-to-queries mapping (collection: queryid2query dictionary)
    """
    lang2qid2query = defaultdict(dict)
    query_languages = {k for k, _ in langpairs_seen}
    for lang in tqdm.tqdm(query_languages):
        print(lang)
        for train_dev in ["dev", "train"]:
            print(f"loading {train_dev} queries")
            if train_dev == "train":
                fName = os.path.join(path_mmarco_googletranslations, "queries", train_dev, f"{short2long[lang]}_queries.{train_dev}.tsv")
            else:
                # multilingual_14l_queries.dev.small.tsv
                fName = os.path.join(path_mmarco_googletranslations, "queries", train_dev, f"{short2long[lang]}_queries.{train_dev}.small.tsv")

            with open(fName, "r", encoding="utf-8") as f:
                for line in f:
                    qid, query = line.split("\t")
                    assert qid not in lang2qid2query
                    lang2qid2query[lang][qid] = query.strip()
    return lang2qid2query


def main():
    lang2qid2query = load_queries()
    lang2did2doc = load_collections()
    train_instances_jsonl = load_instances(filepath=os.path.join(zeroshot_dir, "train_csclir.jsonl"),
                                           total=20_000_000)
    dev_instances_jsonl = load_instances(filepath=os.path.join(zeroshot_dir, "dev_csclir.jsonl"), total=253_433)

    print("writing new instances")
    for qlang, dlang in langpairs_seen:
        tgt_dir = os.path.join(output_dir, qlang + dlang)
        os.makedirs(tgt_dir, exist_ok=True)
        print(f"{qlang}->{dlang}")

        # note: reimers_dev_instances != official msmarco dev, the former is an internal (unused) dev held out dev split
        for name, instances in [("train", train_instances_jsonl), ("dev", dev_instances_jsonl)]:
            print(name)
            lines = []
            for inst in tqdm.tqdm(instances):
                new_inst = {
                    "qid": inst["qid"],
                    "query": lang2qid2query[qlang][inst["qid"]],
                    "pid": inst["pid"],
                    "passage": lang2did2doc[dlang][inst["pid"]],
                    "label": inst["label"]
                }
                lines.append(json.dumps(new_inst) + "\n")

            path = os.path.join(tgt_dir, f"{name}_csclir.jsonl")
            print(f"Writing {path}")
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)


if __name__ == '__main__':
    main()
