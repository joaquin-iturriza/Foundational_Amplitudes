# Foundational Amplitudes

A foundation model for tree- and loop-level **scattering amplitudes** in particle
physics. A single Lorentz-equivariant transformer is trained jointly on many
processes (`ee→WW`, `ee→ttbar`, `ee→uu`, …) and then fine-tuned to new processes
and perturbative orders.

The maintained architecture is a **μP LLoCa** (Lorentz-local) transformer:
`models.lloca.LLOCAMuPTransformer`, wrapped by `wrappers.AmplitudeLLoCaWrapper`.
Particles are encoded by a fixed physical-property table rather than a learned
vocabulary, so adding a new particle or feature never forces retraining the
backbone; coupling order is passed as extra per-particle scalar features, so
mixing perturbative orders needs no model change.

## Install

```bash
conda create -n foundational python=3.11
conda activate foundational
pip install -r requirements.txt
```

`IntrinsicDimDeep/` (intrinsic-dimension diagnostics used during evaluation) is
vendored in-tree; no extra setup is required.

## Quickstart

Training is driven by [Hydra](https://hydra.cc). The entry point is `run.py`;
the default config tree lives under `config/`.

```bash
# train the default μP LLoCa model
python run.py model=lloca training.lr=1e-4

# override any config field from the CLI
python run.py model=lloca data.use_PIDs=false training.dtype=float32
```

A run executes `init_physics → init_data → init_model → train → evaluate → plot`
and writes models, tokenizer, and plots under `runs/<exp_name>/`.

## Data

Two data sources, selected by `data.source`:

- **`files`** (default) — load pre-generated `.npy` datasets from `data/`, each
  row being flat 4-momenta + PDG ids + amplitude.
- **`recipes`** — materialize per-process pools on the fly via `datagen.py`
  (which drives a MadGraph backend in `mg5_pipeline_final.py`). See
  `recipes/pretrain25.yaml` for the spec schema.

## Layout

| Path | What |
|------|------|
| `run.py` | Hydra entry point |
| `experiment.py` | amplitude experiment: physics, data, loss, eval, plots |
| `base_experiment.py` | generic train loop, optimizer/scheduler, μP, checkpointing |
| `models/` | model implementations (μP LLoCa and friends) |
| `wrappers.py` | amplitude wrapper around the backbone |
| `dataset.py` | sparse variable-length event dataset + collation |
| `preprocessing.py` | amplitude/momentum preprocessing |
| `datagen.py`, `mg5_pipeline_final.py` | on-the-fly dataset generation |
| `config/` | Hydra configs |
| `recipes/` | example per-process data recipes |
