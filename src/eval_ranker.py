import pickle
import logging
import tqdm
import os
import json
import argparse
import ir_measures

from src.utils.language_pairs import short2long, short2long_unseen
from transformers import AutoTokenizer
from ir_measures import RR, Measure
from typing import Dict, Union, List, Tuple
from src.utils.helper import load_relass, RelAss, Run


MODEL_HOME = "/path/to/model/home"

ZS = "zeroshot"
FFT = "finetuning"
MCS = "multilingual"
BCS = "bilingual"
WIKICS = "wiki"
LOAD_ONCE = {ZS, WIKICS, MCS}
modes = [ZS, MCS, WIKICS, FFT, BCS]

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g", type=str, required=True)
parser.add_argument("--seed", "-s", type=str, required=True)
parser.add_argument("--mode", "-m", type=str, required=True, choices=modes)
parser.add_argument("--eval_translate_test", "-tt", default=False, action="store_true")
args = parser.parse_args()

selected_mode = args.mode
assert selected_mode in modes

gpu_ = os.environ[f"{args.gpu}"]
seed = args.seed
eval_translate_test = args.eval_translate_test
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_
from sentence_transformers import CrossEncoder

if selected_mode in [ZS, FFT]:
    # there is no translation probability in directory path because models aren't trained on code-switched data
    base_modelpath = os.path.join(MODEL_HOME, f"finetuning/seed_{seed}")
elif selected_mode in [MCS, BCS]:
    # this script is used to evaluate main results, where we set translation probaility to p=0.5
    base_modelpath = os.path.join(MODEL_HOME, f"clwe-cs/seed_{seed}/0.5")
else:
    # translation probability is p=1.0 because we translate all entities.
    assert selected_mode == WIKICS
    base_modelpath = os.path.join(MODEL_HOME,  f"wiki-cs/seed_{seed}/1.0")
assert os.path.exists(base_modelpath)

short2long = {**short2long, **short2long_unseen}
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

path_logfile = os.path.join(base_modelpath, f'log_eval_{selected_mode}.txt')
fh = logging.FileHandler(path_logfile)
fh.setLevel(level=logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.setLevel(level=logging.DEBUG)
logger.info(f"Seed: {seed}")
# logger.info(f"GPU: {args.gpu} ({gpu_})")
logger.info(f"Log file: {path_logfile}")
logger.info(f"Mode: {selected_mode}")

# Path to directory containing
# - "collections_jsonl"-folder (with jsonl files for each language corpus)
# - "translate_test_queries"-folder with mMARCO querry files translated to EN with bilingual dictionaries
# - "translate_test_collections_jsonl"-folder with mMARCO jsonl corpus files translated to EN with bilingual dictionaries
mmarco_google_dir = "/path/to/data_home/google_translations/"
passages_filepath = "/path/to/data_home/top1000.dev"


def load_queries(qlang):
    folder = "translate_test_queries" if eval_translate_test else "queries"
    queries_filepath = os.path.join(mmarco_google_dir, f"{folder}/dev/{short2long[qlang]}_queries.dev.small.tsv")
    queries = {}
    logger.info(f"Loading {qlang} queries ({queries_filepath})")
    with open(queries_filepath, "r", encoding="utf-8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query
    return queries


def load_reranking_data(dlang):
    logger.info("Loading reranking data")

    def load_did2docs():
        result = {}
        folder = "translate_test_collections_jsonl" if eval_translate_test else "collections_jsonl"
        path = mmarco_google_dir + f"{folder}/{short2long[dlang]}"
        files = os.listdir(path)
        logger.info(f"Loading collection ({path})")
        for file in tqdm.tqdm(files):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                for line in f:
                    tmp = json.loads(line)
                    assert tmp["id"] not in result
                    result[tmp["id"]] = tmp["contents"]
        return result

    did2doc = load_did2docs()
    # Read the top 1000 passages that are supposed to be re-ranked
    logger.info("compiling top-1k dev passages")
    passage_candidates = {}
    with open(passages_filepath, "r", encoding="utf-8") as fIn:
        for line in fIn:
            qid, pid, query, passage = line.strip().split("\t")
            if qid not in passage_candidates:
                passage_candidates[qid] = []
            doc = passage if dlang in ["en", "english"] else did2doc[pid]
            passage_candidates[qid].append([pid, doc])
    return passage_candidates


def evaluate(rel_ass: RelAss, run: Run) -> Dict[Measure, Union[float, int]]:
    """
    Compute MRR@10 for current run and relevance assessments
    :param rel_ass: for each query a mapping from document id to relevance
    :param run: for each query a mapping from document id to score
    :return: evaluation result MRR@10
    """
    return ir_measures.calc_aggregate([RR @ 10], rel_ass, run)


def run_retrieval(
        model: CrossEncoder,
        qid2candidates: Dict[str,List[Tuple[str,str]]],
        queries: Dict[str,str],
        qid2pid2score: Dict[str,Dict[str,float]]
):

    relevant_qid = []
    for qid in queries:
        if len(qid2pid2score[qid]) > 0: relevant_qid.append(qid)

    run = {}
    for qid in tqdm.tqdm(relevant_qid):
        query = queries[qid]

        cand = qid2candidates[qid]
        pids = [c[0] for c in cand]
        corpus_sentences = [c[1] for c in cand]

        cross_inp = [[query, sent] for sent in corpus_sentences]
        cross_scores = model.predict(cross_inp).tolist()

        cross_scores_sparse = {}
        for idx, pid in enumerate(pids):
            cross_scores_sparse[pid] = cross_scores[idx]

        sparse_scores = cross_scores_sparse
        run[qid] = {}
        for pid in sparse_scores:
            run[qid][pid] = float(sparse_scores[pid])

    return run


def main():
    tokenizer = AutoTokenizer.from_pretrained("nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large")
    from src.utils.language_pairs import langpairs_seen as language_pairs

    if selected_mode in LOAD_ONCE:
        # - zero-shot (zs)
        # - entity code-switching (ecs)
        # - multilingual code-switching (mcs)
        if selected_mode == ZS:
            modelpath = os.path.join(base_modelpath, "enen", "checkpoint-200000/")
        else:
            modelpath = os.path.join(base_modelpath, "xxxx", "checkpoint-200000/")
        assert os.path.exists(modelpath)

        logger.info(f"Model path: {modelpath}")
        tokenizer.save_pretrained(save_directory=modelpath)
        model = CrossEncoder(modelpath, max_length=512)

    else:
        logger.info("evaluation is slow, so we first verify that all paths exists before starting to evaluate")
        for qlang, dlang in language_pairs:

            if selected_mode == BCS and qlang == dlang == "en":
                continue
            modelpath = os.path.join(base_modelpath, f"{qlang}{dlang}/checkpoint-200000/")
            # special case for full fine-tuning model: load it from folder where models are trained on codemixed data for
            # EN-XX, XX-EN, XX-XX evaluation. All other models are usually in msmarco-minilm/
            if "xx" in [qlang, dlang] and selected_mode == FFT:
                modelpath = os.path.join(
                    # f"/mounts/data/proj/rlitschk/exp/csclir/mmarco-minilm_codemixed/seed_{seed}/0.5",
                    f"/path/to/model_home/clwe-cs/seed_{seed}/0.5",
                    f"{qlang}{dlang}/checkpoint-200000/"
                )
            assert os.path.exists(modelpath), f"{modelpath} does not exist"

    suffix = "_translatetest" if eval_translate_test else ""
    path_runs = os.path.join(base_modelpath, "runs", selected_mode + suffix)
    os.makedirs(path_runs, exist_ok=True)
    logger.info(f"Saving/loading runs to/from {path_runs}")

    lp2eval_result = {}
    reranking_data = {}
    lang2queries = {}
    for qlang, dlang in language_pairs:

        if selected_mode == BCS and qlang == dlang == "en":
            logger.info(f"Skipping EN-EN for mode={selected_mode}")
            lp2eval_result[qlang + dlang] = {RR @ 10: -1}
            continue

        logger.info(f"==== Running {qlang}-{dlang} ====")

        # Only use queries that have at least one relevant passage
        qid2pid2score = load_relass()
        run_file = os.path.join(path_runs, f"{qlang + dlang}.pkl")

        # load evaluation result if it already exists
        if os.path.exists(run_file):
            logger.info(f"Run file found, loading {run_file}")
            with open(run_file, "rb") as f:
                run = pickle.load(f)

        # otherwise, run evaluation
        else:
            logger.info(f"Run file NOT found, starting retrieval+evaluation for {run_file}")
            if selected_mode not in LOAD_ONCE:
                # Load one per language pair:
                # - full finetuning (fft)
                # - bilingual code-switching (bcs)
                modelpath = os.path.join(base_modelpath, f"{qlang}{dlang}/checkpoint-200000/")
                if "xx" in [qlang, dlang] and selected_mode == FFT:
                    modelpath = os.path.join(
                        f"/path/tp/model_home/mmarco-minilm_codemixed/seed_{seed}/0.5",
                        f"{qlang}{dlang}/checkpoint-200000/"
                    )
                logger.info(f"Saving tokenizer to {modelpath}")
                tokenizer.save_pretrained(save_directory=modelpath)
                logger.info(f"Loading model ({modelpath})")
                model = CrossEncoder(modelpath, max_length=512)

            # Load re-ranking data (only once from disk)
            qid2candidates = reranking_data.get(dlang, load_reranking_data(dlang))
            reranking_data[dlang] = qid2candidates

            # Load queries (only once from disk)
            queries = lang2queries.get(qlang, load_queries(qlang))
            lang2queries[qlang] = queries

            logger.info("Queries: {}".format(len(queries)))
            run = run_retrieval(model, qid2candidates, queries, qid2pid2score)

            logger.info(f"Saving run file {run_file}")
            with open(run_file, "wb") as f:
                pickle.dump(run, f)

        eval_result = evaluate(qid2pid2score, run)
        lp2eval_result[qlang + dlang] = eval_result
        logger.info(f"MRR@10: {eval_result[RR @ 10]}")

    header = list(lp2eval_result.keys())
    summary_mrr = ";".join(["MRR@10"] + [str(lp2eval_result[lp][RR @ 10]) for lp in header])
    logger.info(header)
    logger.info(summary_mrr)


if __name__ == '__main__':
    main()
