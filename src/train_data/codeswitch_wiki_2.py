import os
import json
import tqdm
import random

from collections import defaultdict
from src.utils.language_pairs import langpairs_seen, short2long
from src.utils.helper import load_instances


# directory containing csv files (bilingually WIKI-CS'ed instances) from codeswitch_wiki_1.py
bilingual_wiki_cs_input_dir = "/mounts/data/proj/rlitschk/data/ms-marco-reimers/mmarco_entity/translation_prob=1.0"

# target directory where multilingual WIKI-CS output should be stored
# tgt_base_dir = "/mounts/data/proj/rlitschk/data/ms-marco-reimers/mmarco_entity/translation_prob=1.0/xxxx"
multilingual_wiki_cs_output_dir = "/path/to/data/WIKI-CS/translation_prob=1.0/xxxx"

# path to load jsonl EN training and dev data (note: we don't use dev as we train on fixed number of steps)
folder_jsonl_traindev = "/mounts/data/proj/rlitschk/data/ms-marco-reimers"

# path load EN queries and collections (mmarco_googletranslations_dir)
mmarco_googletranslations_dir = "/mounts/data/proj/rlitschk/data/mmarco/google"


random.seed(1)
include_unseen = False
if include_unseen:
    from src.utils.language_pairs import short2long_unseen, mlingpairs_unseen
    short2long = {**short2long, **short2long_unseen}
    langpairs = mlingpairs_unseen


def load_collection(doc_languages):
    print("loading documents")
    lang2did2doc = defaultdict(dict)
    # base_dir = os.path.join(src_dir, "collections_jsonl")
    mmarco_collections_jsonl_dir = os.path.join(mmarco_googletranslations_dir, "collections_jsonl")
    for lang in doc_languages:
        print(lang)
        tgt_dir = os.path.join(mmarco_collections_jsonl_dir, short2long[lang])
        for file in os.listdir(tgt_dir):
            instances = load_instances(filepath=os.path.join(mmarco_collections_jsonl_dir, short2long[lang], file), total=1_000_000)
            assert all(instance["id"] not in lang2did2doc[lang] for instance in instances)
            for instance in instances:
                lang2did2doc[lang][instance["id"]] = instance["contents"].strip()
    return lang2did2doc


def load_wiki_cs_collections():
    print("loading documents")
    lang2did2doc = defaultdict(dict)
    doc_languages = {v for _, v in langpairs_seen}
    doc_languages.remove("en")
    for lang in doc_languages:
        for mode in ["train", "dev"]:
            with open(os.path.join(bilingual_wiki_cs_input_dir, f"{mode}_csclir_{lang}_passage.csv")) as f:
                for line in f:
                    did, doc = line.split(",", maxsplit=1)
                    assert did not in lang2did2doc
                    lang2did2doc[lang][did] = doc.strip()
    return lang2did2doc


def load_queries():
    lang2qid2query = defaultdict(dict)
    query_languages = {k for k, _ in langpairs_seen}
    for lang in tqdm.tqdm(query_languages):
        print(lang)
        for train_dev in ["dev", "train"]:
            print(f"loading {train_dev} queries")
            if lang == "en":
                fName = os.path.join(mmarco_googletranslations_dir, "queries", train_dev, f"{short2long[lang]}_queries.{train_dev}.tsv")
            else:
                fName = os.path.join(bilingual_wiki_cs_input_dir, f"{train_dev}_csclir_{lang}_query.csv")

            with open(fName, "r", encoding="utf-8") as f:
                for line in f:
                    if lang == "en":
                        qid, query = line.split("\t")
                    else:
                        qid, query = line.split(",", maxsplit=1)
                    assert qid not in lang2qid2query
                    lang2qid2query[lang][qid] = query.strip()
    return lang2qid2query


def main():
    lang2qid2query = load_queries()
    lang2did2doc = load_wiki_cs_collections()
    lang2did2doc["en"] = load_collection(["en"])

    train_instances_jsonl = load_instances(filepath=os.path.join(folder_jsonl_traindev, "train_csclir.jsonl"),
                                           total=20_000_000)
    dev_instances_jsonl = load_instances(filepath=os.path.join(folder_jsonl_traindev, "dev_csclir.jsonl"), total=253_433)

    query_languages = list(lang2qid2query.keys())
    doc_languages = list(lang2did2doc.keys())

    print(f"Writing instances to {multilingual_wiki_cs_output_dir}")
    os.makedirs(multilingual_wiki_cs_output_dir, exist_ok=True)
    for name, instances in [("train", train_instances_jsonl), ("dev", dev_instances_jsonl)]:
        print(name)
        multilingual_entitymixed_records = []
        sampled_language_pairs = []
        for inst in tqdm.tqdm(instances):
            sampled_query_lang = query_languages[random.randint(0, len(query_languages) - 1)]
            sampled_doc_lang = doc_languages[random.randint(0, len(doc_languages) - 1)]
            new_inst = {
                "qid": inst["qid"],
                "query": lang2qid2query[sampled_query_lang][inst["qid"]],
                "pid": inst["pid"],
                "passage": lang2did2doc[sampled_doc_lang][inst["pid"]],
                "label": inst["label"],
                # "q-language": sampled_query_lang,
                # "d-language": sampled_doc_lang
            }
            info = {
                "qid": inst["qid"],
                "pid": inst["pid"],
                "q-language": sampled_query_lang,
                "d-language": sampled_doc_lang
            }
            multilingual_entitymixed_records.append(json.dumps(new_inst) + "\n")
            sampled_language_pairs.append(json.dumps(info) + "\n")

        path = os.path.join(multilingual_wiki_cs_output_dir, f"{name}_csclir.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(sampled_language_pairs)

        path = os.path.join(multilingual_wiki_cs_output_dir, f"{name}_csclir.jsonl")
        print(f"Writing {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(multilingual_entitymixed_records)


if __name__ == '__main__':
    main()
