#!/bin/bash
#SBATCH --job-name=zz_lowd_tf32probe
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=zz_lowd_tf32probe_%j.out
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
git log --oneline -1
# the linear-probe trial that failed on H100 (tp3_lowd_lpft_r1_ud_ud_d5 trial_0001), 20 steps, into a probe run dir
python run.py local=none data.color_onehot=true data.coupling_scalars=true data.eval_subsample=10000 data.generation_feature=true data.generation_onehot=true data.internal_mass_pdgs=[23,6,25] data.internal_mass_scalars=true data.mass_from_momenta=false data.offshell_per_event=true data.phase_space_on_pool=true data.preprocess_per_dataset=true data.processes_file=$PROJECT_DIR/recipes/transfer_probe_ud_ud.yaml data.prop_is_massless=true data.require_cache=false data.seed=42 data.signedlog_quantile=0.01 data.source=recipes data.spin_onehot=true data.standardize_props=true data.target_propagator_tchannel=false data.target_propagator_tchannel_max_final=2 data.target_propagators=false data.train_subsample=316 data.use_PIDs=false evaluation.batchsize=16384 evaluation.train_subsample=2000 fine_tune.layer_decay=1.0 fine_tune.linear_probe=true fine_tune.lr_scale=1.0 fine_tune.offshell_stats=parent fine_tune.pretrained_path=$PROJECT_DIR/runs/tp3_ladder_r1/trial_0047/models/model_run0_best.pt fine_tune.target_stats=own model=lloca model.net.num_blocks=8 model.net.num_heads=8 model.particle_encoder_hidden=0 model.use_diagrams=false plot=true seed=42 training.batchsize=16384 training.dtype=float32 training.es_load_best_model=true training.get_ID=false training.loss_aggregation=mean training.regularization=L2 training.save_intermediate=false training.scheduler=CosineAnnealingLR training.sign_head=true training.val_aggregation=geometric_mean training.validate_frac=0.01 use_mlflow=false training.lr=3.5e-03 training.regularization_lambda=1.6e-08 training.cosanneal_warmup_frac=0.11 training.cosanneal_eta_min=7.5e-09 ema=false training.iterations=20 base_dir=$PROJECT_DIR run_dir=$PROJECT_DIR/runs/zz_lowd_tf32probe exp_name=zz_lowd_tf32probe 2>&1 | grep -i "linear probe\|allow_tf32\|Error\|Traceback" | head -20
echo PROBE_DONE
