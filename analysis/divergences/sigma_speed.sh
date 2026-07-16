#!/bin/bash
#SBATCH --job-name=sigspeed
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/sigspeed_%A_%a.out
#SBATCH --error=analysis/divergences/sigspeed_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# Q1 (SPEED of the σ step): read the σ-RANKING convergence curve off a SINGLE
# constant-lr σ-only trajectory, so "how few iters make σ usable as a reweighting
# signal" is answered without the cosine-horizon confound the twostage script
# flagged (a short cosine run is NOT a truncation of a long one).
#
# The σ head is a frozen-trunk 258-param readout, so the whole trajectory is one
# short run. The probe (sigma_speed_probe.rank_metrics, wired into _validate under
# heterosc_rank_curve) records per validation step:
#   ρ_glob  Spearman(σ,|r|) over all events (cross- + within-process ordering)
#   ρ_proc  MEDIAN over processes of within-process Spearman — the HIGH-DIM signal
#           the user cares about (σ ordering events WITHIN a process, where the
#           cross-process scale can't do the ranking for it)
#   slope   reliability slope (secondary, calibration-flavoured)
#   μ-MSE   constant across a σ-only fit (freeze self-check)
# → sigma_rank_curve.json in each run dir; plotted offline by plot_sigma_speed.py.
#
# NOT an HP grid: the array axis is the warm-start MODEL (base22 vs uug) — a non-HP
# data/ckpt ablation axis. lr is ONE inlined constant (0.04, the validated short-
# schedule recipe lr), identical across arms. scheduler=null (constant lr) is the
# deliberate schedule choice for a clean convergence read, not a tuned knob.
#
#   arm 0  base22   THE TARGET. 22-process leave-uug-out MSE foundation. Its 22
#                   per-process MSEs span 4.5 decades — the real high-dim test of
#                   whether σ orders events across AND within processes.
#   arm 1  uug      cheap single-process cross-check (antenna deep-IR finetune).

REC=$MAIN/recipes/pretrain22_heldout_uug.yaml
NAMES=(base22 uug)
CKPTS=(
  $MAIN/runs/pretrain22_heldout_uug/base/models/model_run0_best_2ch.pt
  $MAIN/runs/pretrain22_heldout_uug/ft_deep_antenna/models/model_run0_best_2ch.pt
)
i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; CKPT=${CKPTS[$i]}

if [ "$i" -eq 0 ]; then
  DATA_ARGS=(data.source=recipes data.processes_file="$REC")
else
  DATA_ARGS=(data.source=files data.data_path="$MAIN/data_deep_antenna/"
             'data.dataset=[ee_uug_91-1000GeV_amplitudes]'
             data.preprocess_per_dataset=true
             'data.train_test_val=[0.9, 0.05, 0.05]'
             data.subsample=null)
fi

python run.py \
  exp_name=heterosc_sigspeed run_name=${NAME} \
  "${DATA_ARGS[@]}" \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  model.net.detach_sigma_backbone=false \
  model.net.sigma_after_pool=true \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.reset_output_head=false \
  fine_tune.lr_scale=1.0 fine_tune.layer_decay=1.0 \
  training.loss=HETEROSC \
  training.heterosc_beta=0.0 \
  training.heterosc_sigma_only=true \
  training.heterosc_rank_curve=true \
  training.rank_curve_max_events=50000 \
  training.lr=0.04 \
  training.iterations=800 training.batchsize=16384 \
  evaluation.batchsize=16384 \
  training.validate_frac=0.0 training.validate_every_n_steps=15 \
  training.es_patience=100000 \
  training.clip_grad_norm=5 \
  training.regularization=L2 training.regularization_lambda=9.892346e-07 \
  training.scheduler=null \
  plot=true save=true

echo "DONE ${NAME}"
