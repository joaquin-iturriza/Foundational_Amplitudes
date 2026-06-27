"""Measure training FLOPs/step (forward+backward) for each architecture on the
real 25-process recipe pipeline, so the comparison runs can be matched on compute.

Builds the actual AmplitudeExperiment (same data recipe, same batch=1024, same
per-step loss path `_batch_loss`), grabs one real batch, and counts FLOPs of the
forward+backward with torch's dispatch-level FlopCounterMode (architecture-agnostic
— it sees the geometric EquiLinear/attention as the bmm/sdpa they decompose into).

A small train_subsample keeps data loading fast: FLOPs/step depends on the batch's
event sizes (real, preserved), not on how many events sit in the pool.

Usage (one GPU job loops over the three):
    python compare_models/profile_flops.py lloca   >> flops.txt
    python compare_models/profile_flops.py lgatr    >> flops.txt
    python compare_models/profile_flops.py slim     >> flops.txt
Prints a line:  FLOPS_STEP <model> <flops>
"""
import sys

PROJECT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, PROJECT)  # script lives in compare_models/; put project root on path

import torch
from hydra import compose, initialize_config_dir
from torch.utils.flop_counter import FlopCounterMode

from experiment import AmplitudeExperiment

# Shared recipe / training recipe (mirrors sweep_config_jeanzay_pretrain25 fixed_params),
# with a tiny train_subsample so init_data is fast — irrelevant to FLOPs/step.
SHARED = [
    "local=none",
    "data.source=recipes",
    f"data.processes_file={PROJECT}/recipes/pretrain25.yaml",
    f"data.data_path={PROJECT}/data/",
    "data.preprocess_per_dataset=true",
    "data.require_cache=true",
    "data.train_subsample=4000",
    "data.eval_subsample=2000",
    "data.seed=42",
    "seed=42",
    "training.batchsize=1024",
    "evaluation.batchsize=8192",
    "training.iterations=10",
    "training.validate_frac=0.5",
    "training.loss_aggregation=geometric_mean",
    "training.regularization=L2",
    "training.regularization_lambda=1e-8",
    "training.scheduler=CosineAnnealingLR",
    "training.dtype=float32",
    "plot=false",
    "train=false",            # build only; we drive one step by hand
    "use_mlflow=false",
]

# Per-model overrides: matched to ~1.61M params (LLoCa default).
MODELS = {
    "lloca": ["model=lloca", "model.net.num_blocks=8", "model.net.num_heads=8"],
    "lgatr": ["model=lgatr_mup", "model.net.num_blocks=8",
              "model.net.hidden_mv_channels=22", "model.net.hidden_s_channels=22"],
    "slim":  ["model=lgatr_slim", "model.net.num_blocks=8",
              "model.net.hidden_v_channels=52", "model.net.hidden_s_channels=104"],
}


def main():
    which = sys.argv[1]
    torch.set_default_dtype(torch.float32)
    run_dir = f"{PROJECT}/compare_models/_probe_{which}"  # _init_directory creates it
    with initialize_config_dir(config_dir=f"{PROJECT}/config", version_base=None):
        cfg = compose(config_name="amplitudes",
                      overrides=SHARED + MODELS[which] + [f"run_dir={run_dir}"])

    exp = AmplitudeExperiment(cfg)
    # Replicate full_run()'s init sequence up to (but not including) train(), so
    # _batch_loss has the loader, loss fn, regularization, and mom mean/std it needs.
    exp._init()                  # warm_start, run dir/name, logger, backend (sets device)
    exp.init_physics()
    exp.init_geometric_algebra()
    exp.init_data()
    exp._init_dataloader()
    exp.init_model()
    exp._init_loss()
    exp._init_regularization()
    exp.model.train()

    batch = next(iter(exp.train_loader))

    # warm-up step (lazy init / cudnn autotune) outside the counter
    loss, _, _ = exp._batch_loss(batch)
    loss.backward()
    exp.model.zero_grad(set_to_none=True)

    fcm = FlopCounterMode(display=False)
    with fcm:
        loss, _, _ = exp._batch_loss(batch)
        loss.backward()
    flops = fcm.get_total_flops()
    exp.model.zero_grad(set_to_none=True)

    nparams = sum(p.numel() for p in exp.model.parameters())
    print(f"FLOPS_STEP {which} {flops} params {nparams} "
          f"batch_particles {int(batch[4][-1])}", flush=True)

    # Wall-clock per iteration (fwd+bwd; optimizer step is O(params), negligible vs
    # the ~TFLOP fwd+bwd). This is the honest "GPU time/step" — FLOPs alone miss that
    # the geometric einsums may be far less hardware-efficient than dense matmuls.
    import time
    cuda = torch.cuda.is_available()
    for _ in range(3):  # warmup (cudnn autotune, lazy init)
        loss, _, _ = exp._batch_loss(batch); loss.backward(); exp.model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize()
    NIT = 20
    t0 = time.time()
    for _ in range(NIT):
        loss, _, _ = exp._batch_loss(batch); loss.backward(); exp.model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize()
    ms = (time.time() - t0) / NIT * 1000
    print(f"MS_STEP {which} {ms:.1f} (fwd+bwd, batch_particles {int(batch[4][-1])})", flush=True)


if __name__ == "__main__":
    main()
