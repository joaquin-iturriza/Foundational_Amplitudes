"""The solo references' values (collected in solo_b1k_inspect.json): per sweep, the best trial's best
validation MSE, the DyHPO result (the minimum of the trial's val_loss_no_reg over its validations;
CLAUDE.md, Reported values). The two signed pools come from their rerun, sweeps/solob1kv_t<S>_<p>:
in sweeps/solob1k_t<S>_<p> the single-process validation of a signed pool added the sign head's
cross-entropy (fixed in experiment._batch_loss_lloca, 6939d77), so their curves, checkpoints and
DyHPO results there are MSE + BCE and are not used."""
import json, os
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solo_b1k_inspect.json")
SIGNED = {"udbar_Wgg_nlo", "uubar_ddbara_nlo"}
SWEEP = lambda p, S: f"solob1kv_t{S}_{p}" if p in SIGNED else f"solob1k_t{S}_{p}"

def solo_mse():
    """{"<process>|<steps>": the sweep's best validation MSE}"""
    return {k: min(t["val_loss"] for t in v) for k, v in json.load(open(JSON)).items() if v}
