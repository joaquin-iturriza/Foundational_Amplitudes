"""The solo references' values (sweeps/solob1k_t<S>_<process>, collected in solo_b1k_inspect.json): per
sweep, the best trial's best validation MSE (the DyHPO result, the minimum of its validation curve).
Except on the two signed pools: there the single-process validation added the sign head's
cross-entropy (fixed in experiment._batch_loss_lloca), so the curve and the DyHPO result are MSE + BCE
and the only MSE on record is the end-of-training one ("MSE (prepd) val"); those sweeps are provisional
until rerun with the fix."""
import json, os
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solo_b1k_inspect.json")
SIGNED = {"udbar_Wgg_nlo", "uubar_ddbara_nlo"}

def solo_mse():
    """{"<process>|<steps>": the sweep's best validation MSE}"""
    out = {}
    for k, v in json.load(open(JSON)).items():
        if k.split("|")[0] in SIGNED:
            v = [t["final_val"] for t in v if "final_val" in t]
        else:
            v = [t["val_loss"] for t in v]
        if v: out[k] = min(v)
    return out
