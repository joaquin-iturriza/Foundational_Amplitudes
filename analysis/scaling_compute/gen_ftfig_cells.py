"""Generate + (optionally) submit the fine-tune cells for the ft416/ft352 compute-scan
curves and the 1h feature-ladder markers.

One sbatch job per (family, target, D) chains that cell-column's t-grid as sequential
run.py fine-tunes (single-dataset target, files source), each writing the
{val_loss, test_loss, traintime_hours} JSON run.py emits (evaluate=true +
training.result_path) into sweeps/cscan_<fam>_D<D>_<key>_t<t>/results/ — the exact
layout build_compute_scan_plot*.py already consumes, so the plots pick the new
curves up with only a CURVES entry.

HPs per cell come from analysis/scaling_compute/ftfig_cell_hps.json (ft25 best per
cell; ft8/ft25 midpoint where they agreed — see the harvest commit). The fixed
params mirror the cscan_ft25 cells (training.lr=0.008322839, bs 16384, ttv
[0.7,0.2,0.1], validate_frac 0.01). Families:
  ft416raw  <- _ftfig_pre_raw416   (pt25-era encoding flags)
  ft416best <- _ftfig_pre_best416  (adopted-candidate flags, amp_orders [[1,0]])
  ft352lo   <- _ftfig_pre_lo352    (same flags, LO-only pretrain)
Ladder (one job): raw1h/rung2_1h/rung3_1h/best1h fine-tuned at D=100k, t=6193 only.
"""
import json, os, stat, sys

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
HPS = json.load(open(f"{ROOT}/analysis/scaling_compute/ftfig_cell_hps.json"))
JOBDIR = f"{ROOT}/sweeps/ftfig_jobs"
os.makedirs(JOBDIR, exist_ok=True)

TARGETS = {"eeuunlovirte4": "ee_uu_nlo_virt_e4", "eettbarnlovirte4": "ee_ttbar_nlo_virt_e4"}
TGRID = {"1k":  [362, 1087, 3624, 10872, 36240, 72480],
         "10k": [36, 109, 362, 1087, 3624, 10872, 28992],
         "100k": [8, 23, 77, 232, 774, 2323, 6193],
         "1M":  [8, 23, 77, 232, 774, 2323, 6193]}
SUB = {"1k": 1000, "10k": 10000, "100k": 100000, "1M": 1000000}

RAW_FLAGS = ("data.spin_onehot=false data.color_onehot=false data.prop_is_massless=false "
             "data.standardize_props=false data.mass_from_momenta=false "
             "data.coupling_scalars=false data.internal_mass_scalars=false "
             "data.offshell_per_event=false")
BEST_FLAGS = ("data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true "
              "data.standardize_props=true data.mass_from_momenta=true "
              "data.coupling_scalars=true data.internal_mass_scalars=true "
              "data.offshell_per_event=true data.internal_mass_pdgs=[23,6,25] "
              "data.amp_orders=[[1,0]]")
FAMS = {"ft416raw":  ("raw416",  RAW_FLAGS),
        "ft416best": ("best416", BEST_FLAGS),
        "ft352lo":   ("lo352",   BEST_FLAGS)}
LADDER = {"ftraw1h": ("raw1h", RAW_FLAGS), "ftrung2": ("rung2_1h",
          "data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true "
          "data.standardize_props=true data.mass_from_momenta=false data.coupling_scalars=false "
          "data.internal_mass_scalars=false data.offshell_per_event=false"),
          "ftrung3": ("rung3_1h",
          "data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true "
          "data.standardize_props=true data.mass_from_momenta=true data.coupling_scalars=true "
          "data.internal_mass_scalars=false data.offshell_per_event=false"),
          "ftbest1h": ("best1h", BEST_FLAGS)}

HEADER = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time={hours}:00:00
#SBATCH --output={jobdir}/{name}_%j.out
#SBATCH --error={jobdir}/{name}_%j.err
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd {root}
"""

def run_cmd(fam, arm, flags, key, D, t, hp):
    cell = f"{ROOT}/sweeps/cscan_{fam}_D{D}_{key}_t{t:05d}"
    hpargs = (f"fine_tune.lr_scale={hp['fine_tune.lr_scale']:.6g} "
              f"fine_tune.layer_decay={hp['fine_tune.layer_decay']:.6g} "
              f"training.regularization_lambda={hp['training.regularization_lambda']:.6g} "
              f"training.cosanneal_warmup_frac={hp['training.cosanneal_warmup_frac']:.6g} "
              f"training.cosanneal_eta_min={hp['training.cosanneal_eta_min']:.6g}")
    return f"""mkdir -p {cell}/results
python run.py model=lloca local=none \\
  model.net.num_heads=8 model.net.num_blocks=8 \\
  data.data_path={ROOT}/data/ data.dataset=[{TARGETS[key]}] \\
  data.subsample={SUB[D]} data.train_test_val=[0.7,0.2,0.1] \\
  data.preprocess_per_dataset=true data.seed=42 seed=42 data.use_PIDs=false \\
  {flags} \\
  model.use_diagrams=false model.particle_encoder_hidden=0 \\
  fine_tune.pretrained_path={ROOT}/compare_models/_ftfig_pre_{arm}/models/model_run0_best.pt.gz \\
  {hpargs} \\
  training.lr=0.008322839 training.batchsize=16384 evaluation.batchsize=8192 \\
  training.loss_aggregation=geometric_mean training.regularization=L2 \\
  training.scheduler=CosineAnnealingLR training.dtype=float32 \\
  training.validate_frac=0.01 training.es_load_best_model=false \\
  training.get_ID=false use_mlflow=false evaluate=true plot=true \\
  training.iterations={t} \\
  training.result_path={cell}/results/hp0000_t{t}_$(date +%s).json \\
  exp_name={fam}_{key}_t{t} run_dir={cell}/run \\
  && echo "OK {fam} {key} D{D} t{t}" || echo "FAIL {fam} {key} D{D} t{t}"
"""

scripts = []
for fam, (arm, flags) in FAMS.items():
    for key in TARGETS:
        for D, ts in TGRID.items():
            name = f"{fam}_{key[:4]}{'tt' if 'tt' in key else ''}_D{D}"
            hours = 3 if D in ("1k", "10k") else 2
            body = HEADER.format(name=name, hours=hours, jobdir=JOBDIR, root=ROOT)
            n = 0
            for t in ts:
                hp = HPS.get(f"{key}|{D}|{t}")
                if hp is None:
                    print(f"  !! no HPs for {key}|{D}|{t} — skipped"); continue
                body += run_cmd(fam, arm, flags, key, D, t, hp); n += 1
            if n:
                p = f"{JOBDIR}/{name}.sh"
                open(p, "w").write(body + f'echo "===== {name} DONE ====="\n')
                os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
                scripts.append(p)

# ladder: one job, D=100k, t=6193, both targets
body = HEADER.format(name="ftladder", hours=3, jobdir=JOBDIR, root=ROOT)
for fam, (arm, flags) in LADDER.items():
    for key in TARGETS:
        hp = HPS[f"{key}|100k|6193"]
        body += run_cmd(fam, arm, flags, key, "100k", 6193, hp)
p = f"{JOBDIR}/ftladder.sh"
open(p, "w").write(body + 'echo "===== ftladder DONE ====="\n')
os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
scripts.append(p)

print(f"generated {len(scripts)} job scripts in {JOBDIR}")
for s in scripts: print(" ", os.path.basename(s))
