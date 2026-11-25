"""Per family of runs (trim_plan.tsv's study mapping, build_inventory.runs): GB of model files, how many of them D133
allows (study CONCLUDED, trained under 5 h, not protected, no model loader), the study and its status, training hours.
Read-only; prints TSV sorted by model GB.
    python analysis/storage/model_groups.py RUNS.jsonl [RUNS.jsonl ...]
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_inventory import load_map, runs   # noqa: E402

inv = runs(sys.argv[1:], load_map())
print("family_glob\tstatus\tstudy\truns\tGB\tmodel_GB\tprotected_model_GB\truns_over_5h\ttrain_h_median\ttrain_h_max\tloaded_by")
for g in sorted(inv["by_study"], key=lambda g: -g["model_GB"]):
    print("\t".join(str(g[k]) for k in ("family_glob", "status", "study", "runs", "GB", "model_GB", "protected_model_GB",
                                        "runs_over_5h", "train_h_median", "train_h_max", "models_loaded_by")))
c = inv["candidate_sets"]["under_5h"]
print(f"#D133_CANDIDATES\t{c['runs']} runs\t{c['model_GB_freed']} GB\t{c['GB_by_study']}")
print(f"#MODEL_GB_BY_ROOT\t{inv['model_GB_by_root']}\t#TOTAL_GB_BY_ROOT\t{inv['total_GB_by_root']}")
