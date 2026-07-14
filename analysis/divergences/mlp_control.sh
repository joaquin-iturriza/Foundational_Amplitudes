#!/bin/bash
#SBATCH --job-name=mlp_ctrl
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/mlpctrl_%A_%a.out
#SBATCH --error=analysis/divergences/mlpctrl_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# THE REFERENCE'S ACTUAL CONFIGURATION. heidelberg-hepml/amplitude_DSI wires HETEROSC ONLY into
# the MLP (mup_mlp.py / mlp.py), whose readout is already PER-EVENT -- it never wires it into the
# LLoCa transformer at all. Our losses.py het loss is BYTE-IDENTICAL to theirs.
#
# On LLoCa, het gives reliability slope ~1.4 (sigma under-dispersed, hard tail under-predicted
# ~2.5x) and mu-MSE 14x worse than MSE -- and the sigma-pool fix changed NOTHING (slope 1.42 vs
# 1.44). So the remaining structural difference from the validated setup is the ARCHITECTURE:
# per-particle + pooling (LLoCa) vs one output per event (MLP).
#
# This runs the validated combination in THIS codebase, on THIS data:
#   arm 0 mlp_mse : MuMLP + MSE       (accuracy reference for the MLP)
#   arm 1 mlp_het : MuMLP + HETEROSC  (the config the reference validated)
# Read-out: het slope ~1.0 AND accuracy ~ the MLP's MSE arm => the loss is fine and the problem
#   is LLoCa-specific (pooled per-particle sigma). het slope ~1.4 here TOO => the pathology
#   follows the loss/data, not the architecture.
# INPUT FEATURES: the reference's MLP config feeds trafos = {fvs_standardized: [standardization],
#   invs: [invs, log, standardization]} -- i.e. LOG of the Lorentz INVARIANTS s_ij. Those are the
#   pole variables: near an IR pole |M|^2 ~ 1/s_ij, so log|M|^2 ~ -log(s_ij) and the target is close
#   to LINEAR in the features. Our LLoCa gets raw boosted four-momenta with no explicit invariants
#   (trafos: [] in our config). This is a completely different input basis and is reproduced here.
# NOTE: slope ~1 is only meaningful if mu is ACTUALLY FIT -- an undertrained model trivially gets
# slope ~1 (we measured that trade-off). Hence the MSE arm, to bound what the MLP can reach.
NAMES=(mlp_mse mlp_het)
LOSSES=(MSE     HETEROSC)
i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; LOSS=${LOSSES[$i]}

# mup_mlp.yaml hardcodes net.loss: "MSE" (unlike lloca.yaml's ${training.loss}), so set it too.
EXTRA="model.net.loss=${LOSS}"
[ "$LOSS" = "HETEROSC" ] && EXTRA="$EXTRA training.heterosc_beta=0.0"   # plain NLL = the reference

python run.py \
  exp_name=heterosc_mlpctrl run_name=${NAME} \
  model=mup_mlp \
  data.source=files data.data_path="$MAIN/data_deep_antenna/" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true data.include_permsym=false \
  '+data.trafos={fvs_standardized: [standardization], invs: [invs, log, standardization]}' \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  training.loss=${LOSS} \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  training.clip_grad_norm=5 \
  ${EXTRA} \
  plot=true save=true

echo "DONE ${NAME}"
