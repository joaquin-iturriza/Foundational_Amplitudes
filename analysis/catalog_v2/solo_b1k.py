"""The solo references' values (sweeps/solob1k_t<S>_<process>, collected in solo_b1k_inspect.json): per
sweep, the trial with the lowest post-training validation MSE ("MSE (prepd) val" of its log, the
model at the end of training, as the joint runs' last validation). Not the DyHPO result: the
single-process validation added the sign head's cross-entropy on a signed pool (fixed in
experiment._batch_loss_lloca), so those values were MSE + BCE, 6-10x the MSE of the two signed
one-loop references."""
import json, os
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solo_b1k_inspect.json")

def solo_mse():
    """{"<process>|<steps>": validation MSE of the sweep's best trial}"""
    out = {}
    for k, v in json.load(open(JSON)).items():
        v = [t for t in v if "final_val" in t]
        if v: out[k] = min(t["final_val"] for t in v)
    return out
