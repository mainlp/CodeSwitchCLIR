import os
import pickle
import ir_measures
import tqdm

from scipy.stats import ttest_rel
from ir_measures import RR
from src.utils.language_pairs import langpairs_seen
from typing import Dict
from src.utils.helper import load_relass, RelAss, Run


langpairs = langpairs_seen
MODEL_HOME = "/path/to/model_home"

def print_results(langpair2model2pvalue: Dict[str,Dict])->None:
    header = [ql + dl for ql, dl in langpairs_seen]
    rows = [header]
    for mode in langpair2model2pvalue[header[0]]:
        rows.append([mode] + [langpair2model2pvalue[lp][mode] for lp in langpair2model2pvalue])
    for row in rows:
        print(";".join([str(elem) for elem in row]))


def test_significance(relass: RelAss, run_model: Run, run_zershot: Run):
    """
    Run paired t-test, comparing RR@10 values of model against RR@10 values of zero-shot model.
    :param relass: relevance assessments
    :param run_model: RR@10 values of model to test if difference is significant
    :param run_zershot: RR@10 values of refernce (zero-shot) model
    :return: 
    """
    rr_values_zeroshot = {m.query_id: m.value for m in ir_measures.iter_calc([RR @ 10], relass, run_zershot)}
    rr_values_bcs = {m.query_id: m.value for m in ir_measures.iter_calc([RR @ 10], relass, run_model)}
    qids = list(rr_values_zeroshot.keys())
    return ttest_rel([rr_values_zeroshot[v] for v in qids], [rr_values_bcs[v] for v in qids])


def select_model(dlang:str, qlang:str, relass:RelAss, base_path_runs:str):
    """
    Compare the best seed of each approach. model_select evaluates all seeds and returns the results of the best seed.
    :param dlang: document languages
    :param qlang: query language
    :param relass: relevance assessments
    :param base_path_runs: folder containing run files
    :return: loaded run file, best seed, RR@10 values of best seed (used for significance testing)
    """
    best_mrr10 = -1
    best_seed = -1
    best_rr_values_zeroshot = None
    for seed in range(1, 4):
        path_runs  = base_path_runs.replace("##", str(seed))
        run_file = os.path.join(path_runs, f"{qlang + dlang}.pkl")
        with open(run_file, "rb") as f:
            run = pickle.load(f)
        rr_values = {m.query_id: m.value for m in ir_measures.iter_calc([RR @ 10], relass, run)}
        mrr10 = sum(rr_values.values()) / len(rr_values)
        if mrr10 > best_mrr10:
            best_mrr10 = mrr10
            best_seed = seed
            best_rr_values_zeroshot = rr_values
    return run, best_seed, best_rr_values_zeroshot


def main():
    ZS = "zeroshot"
    FFT = "finetuning"
    MCS = "multilingual"
    WIKICS = "wiki"
    BCS = "bilingual"
    modes = [ZS, MCS, WIKICS, FFT, BCS]
    relass = load_relass()

    langpair2model2pvalue = {}
    for qlang, dlang in langpairs_seen:
        print(f"Running {qlang}-{dlang}")
        # load (best) Zero-shot (load all seeds and select best run)
        base_modelpath = os.path.join(MODEL_HOME, f"finetuning/seed_##/runs/zeroshot")
        run_zershot, best_seed_zeroshot, best_rr_values_zeroshot = select_model(dlang, qlang, relass, base_modelpath)

        model2pvalue = {}
        # skip zero-shot (first elem in modes) since it's our t-test reference
        for selected_mode in tqdm.tqdm(modes[1:]):
            print(selected_mode)
            if selected_mode == FFT:
                base_modelpath = os.path.join(MODEL_HOME, f"finetuning/seed_##/runs/{selected_mode}")
            elif selected_mode == WIKICS:
                base_modelpath = os.path.join(MODEL_HOME, f"wiki-cs/seed_##/1.0/runs/{selected_mode}")
            else:
                base_modelpath = os.path.join(MODEL_HOME, f"clwe-cs/seed_##/0.5/runs/{selected_mode}")
            run_ref, best_seed_ref, best_rr_values_ref = select_model(dlang, qlang, relass, base_modelpath)
            result = test_significance(relass, run_ref, run_zershot)
            model2pvalue[selected_mode] = result.pvalue

        if qlang == "xx" or dlang == "xx":
            # we currently don't have results for translate test in MLIR
            model2pvalue[ZS+"_translatetest"] = "-"
            model2pvalue[MCS + "_translatetest"] = "-"
            langpair2model2pvalue[qlang + dlang] = model2pvalue
            print_results(langpair2model2pvalue)
            continue

        # evaluate Zero-shot Translate Test
        base_modelpath = os.path.join(MODEL_HOME, f"zeroshot/seed_##/runs/{ZS}_translatetest")
        run_ref, best_seed_ref, best_rr_values_ref = select_model(dlang, qlang, relass, base_modelpath)
        result = test_significance(relass, run_ref, run_zershot)
        model2pvalue[ZS+"_translatetest"] = result.pvalue

        # evaluate ML-CS Translate Test
        base_modelpath = os.path.join(MODEL_HOME, f"clwe-cs/seed_##/0.5/runs/{MCS}_translatetest")
        run_ref, best_seed_ref, best_rr_values_ref = select_model(dlang, qlang, relass, base_modelpath)
        result = test_significance(relass, run_ref, run_zershot)
        model2pvalue[MCS + "_translatetest"] = result.pvalue

        langpair2model2pvalue[qlang+dlang] = model2pvalue
        print_results(langpair2model2pvalue)

    print_results(langpair2model2pvalue)


if __name__ == '__main__':
    main()
