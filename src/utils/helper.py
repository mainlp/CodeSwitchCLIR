import json
import tqdm

from typing import Dict, List
from collections import defaultdict


RelAss = Dict[str, Dict[str, float]] # qid-did-relevance values
Run = Dict[str, Dict[str, float]] # qid-did-RR@10 values


def chunk(lst: List, n: int):
  """ Yield successive n-sized chunks from lst. (Credit: https://stackoverflow.com/a/312464) """
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

# Path to qrels.dev.small.tsv or $DATA_HOME
# qrels_filepath = "/mounts/data/proj/rlitschk/data/mmarco/qrels.dev.small.tsv"
qrels_filepath = ""
def load_relass()->RelAss:
    """
    Load Relevance assessments
    :return: qid2pid2relevance_score
    """
    relevant_docs = defaultdict(lambda: defaultdict(int))
    print(f"Loading qlres ({qrels_filepath})")
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, score = line.strip().split()
            score = int(score)
            if score > 0:
                relevant_docs[qid][pid] = score
    return relevant_docs


def load_instances(filepath: str, total: int=None) -> List[Dict]:
    """
    Return lines/instances of jsonl file (train file, dev file)
    :param filepath: pointing to jsonl file
    :param total: number of instances (for tqdm)
    :return: list of loaded jsonl records (list of dictionaries)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        instances = []
        for line in tqdm.tqdm(f, total=total):
            instances.append(json.loads(line))
    return instances
