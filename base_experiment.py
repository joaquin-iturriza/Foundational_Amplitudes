import numpy as np
import random
import torch
import torch.nn as nn
import math
import datetime
import pickle
import os, time
import io
import zipfile
import logging
import pandas as pd
import glob
import gzip
import copy
import shutil
import matplotlib.pyplot as plt
from torch.utils.flop_counter import FlopCounterMode

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from omegaconf import OmegaConf, open_dict, errors
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
#import mlflow
from torch_ema import ExponentialMovingAverage

from misc import get_device, flatten_dict
import logger
from logger import LOGGER, MEMORY_HANDLER, FORMATTER
from plots import plot_gradients, plot_weights
#from mlflow_util import log_mlflow
from misc import cosine_warmup_scheduler

from lion_pytorch import Lion
import schedulefree

from IntrinsicDimDeep.get_dim import get_intrinsic_dim

import lgatr.primitives.attention
import lgatr.layers.linear
import lgatr.layers.mlp.geometric_bilinears
import lgatr.layers.mlp.mlp
import lgatr.primitives.linear
from lgatr.layers.mlp.config import MLPConfig
from lgatr.layers.attention.config import SelfAttentionConfig

from mup import MuAdam, MuAdamW  # μP optimizers
from lloca.mup import finalize as mup_finalize  # μP base shapes are now self-contained
                                                # in the backbones; we only finalize the
                                                # parameters living outside them.

cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)

def _torch_load(path, **kwargs):
    """Load a PyTorch checkpoint, transparently decompressing .pt.gz if needed."""
    if path.endswith('.gz') and os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, **kwargs)
    if os.path.exists(path):
        return torch.load(path, **kwargs)
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rb') as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, **kwargs)
    raise FileNotFoundError(path)


# set to 'True' to debug autograd issues (slows down code)
torch.autograd.set_detect_anomaly(False)
MIN_STEP_SKIP = 1000


class BaseExperiment:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        # pass all exceptions to the logger
        try:
            self.run_mlflow()
        except errors.ConfigAttributeError:
            LOGGER.exception(
                "Tried to access key that is not specified in the config files"
            )
        except:
            LOGGER.exception("Exiting with error")

        # print buffered logger messages if failed
        if not logger.LOGGING_INITIALIZED:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            MEMORY_HANDLER.setTarget(stream_handler)
            MEMORY_HANDLER.close()

    def is_main_process(self):
        return (not dist.is_initialized()) or dist.get_rank() == 0
    
    def run_mlflow(self):
        experiment_id, run_name = self._init()
        git_hash = os.popen("git rev-parse HEAD").read().strip()
        LOGGER.info(
            f"### Starting experiment {self.cfg.exp_name}/{run_name} (mlflowid={experiment_id}) (jobid={self.cfg.jobid}) (git_hash={git_hash} ###"
        )
        if self.cfg.use_mlflow:
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                self.full_run()
        else:
            # dont use mlflow
            self.full_run()

    def full_run(self):
        # implement all ml boilerplate as private methods (_name)
        t0 = time.time()

        # save config
        LOGGER.debug(OmegaConf.to_yaml(self.cfg))
        self._save_config("config.yaml", to_mlflow=True)
        self._save_config(f"config_{self.cfg.run_idx}.yaml")

        self.init_physics()
        self.init_geometric_algebra()
        self.init_data()
        self._init_dataloader()
        self.init_model()
        self._init_loss()
        self._init_regularization()
        if self.device == torch.device("cuda"):
            free_mem, total_mem = torch.cuda.mem_get_info()
            LOGGER.info(f"Available VRAM: {free_mem / 1024**2:.2f} MB")
            LOGGER.info(f"Total VRAM: {total_mem / 1024**2:.2f} MB")
            n_params = sum(p.numel() for p in self.model.parameters())
            param_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            model_alloc_mb = torch.cuda.memory_allocated() / 1024**2
            LOGGER.info(
                f"Model (uncompressed): {n_params:,} parameters | "
                f"param tensor memory: {param_mb:.2f} MB | "
                f"CUDA allocated: {model_alloc_mb:.2f} MB"
            )
            torch.cuda.reset_peak_memory_stats()

        if self.cfg.count_flops:
            self.count_flops(dataloader=self.train_loader)

        if self.cfg.train:
            self._init_ewc()
            self._init_optimizer()
            self._init_scheduler()
            self.train()
            self._save_model()

        if self.device == torch.device("cuda"):
            peak = torch.cuda.max_memory_allocated()
            LOGGER.info(f"Peak VRAM used: {peak / 1024**2:.2f} MB")

        if self.is_main_process():
            # Only run a standalone evaluate() when we didn't train (eval-only mode),
            # or when save_intermediate=False meaning _evals was never called inside
            # train() and we need a final evaluation here.
            if self.cfg.evaluate and (not self.cfg.train or not self.cfg.training.save_intermediate):
                self.evaluate()
                result_path = self.cfg.training.get("result_path", None)
                if result_path and os.path.exists(result_path) and hasattr(self, "results_test"):
                    try:
                        import json as _json
                        with open(result_path) as _f:
                            _result = _json.load(_f)
                        _combined_key = next(iter(self.results_test))
                        _result["test_loss"] = float(
                            self.results_test[_combined_key]["preprocessed"]["mse"]
                        )
                        with open(result_path, "w") as _f:
                            _json.dump(_result, _f)
                    except Exception:
                        pass

            if self.cfg.plot and self.cfg.save:
                self.plot()

        if self.device == torch.device("cuda"):
            max_used = torch.cuda.max_memory_allocated()
            max_total = torch.cuda.mem_get_info()[1]
            LOGGER.info(
                f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
            )
        if self.is_main_process():
            self.compress_models()
        dt = time.time() - t0
        LOGGER.info(
            f"Finished experiment {self.cfg.exp_name}/{self.cfg.run_name} after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )
    
    def init_geometric_algebra(self):
        lgatr.primitives.linear.USE_FULLY_CONNECTED_SUBGROUP = (
            self.cfg.ga_settings.use_fully_connected_subgroup
        )
        if self.cfg.ga_settings.use_fully_connected_subgroup:
            lgatr.layers.linear.MIX_MVPSEUDOSCALAR_INTO_SCALAR = (
                self.cfg.ga_settings.mix_mvpseudoscalar_into_scalar
            )
        else:
            lgatr.layers.linear.NUM_PIN_LINEAR_BASIS_ELEMENTS = 5
            if self.cfg.ga_settings.mix_mvpseudoscalar_into_scalar:
                LOGGER.warning(
                    "Mixing mvpseudoscalar into scalar is only possible if "
                    "ga_settings.use_fully_connected_subgroup=True"
                )
            lgatr.layers.linear.MIX_MVPSEUDOSCALAR_INTO_SCALAR = False
        lgatr.layers.mlp.mlp.USE_GEOMETRIC_PRODUCT = (
            self.cfg.ga_settings.use_geometric_product
        )
        lgatr.layers.mlp.geometric_bilinears.ZERO_BIVECTOR = (
            self.cfg.ga_settings.zero_bivector
        )

    def _post_instantiate_model(self, model):
        """Hook called immediately after every model instantiation (main, base, delta).
        Override in subclasses to add custom modules (e.g. particle_encoder) so they
        exist before MuP base shapes are computed and before warm-start loading."""
        pass

    def init_model(self):
        # === instantiate model ===
        # `loss` is a (largely vestigial) net field that only some backbones declare;
        # propagate the training loss into it only when the net config has the key.
        # The lloca transformer declares it (and its ctor accepts it); the lgatr
        # backbones don't, and their constructors don't take a `loss` kwarg.
        if "loss" in self.cfg.model.net:
            self.cfg.model.net.loss = self.cfg.training.loss
        self.model = instantiate(self.cfg.model)
        self._post_instantiate_model(self.model)

        # === μP setup ===
        # μP base shapes are computed inside the backbone's __init__ (self-contained,
        # see lloca.mup): the width-bearing backbone already has its infshapes set by
        # the time `instantiate` returns. We only need to mark the parameters that live
        # *outside* that backbone (e.g. particle_encoder added in _post_instantiate_model,
        # or the framesnet) as standard parametrization, which finalize() does. This
        # replaces the old base/delta-model + make_base_shapes/set_base_shapes + .bsh-file
        # dance that used to live here. Reloading a checkpoint needs no special handling:
        # reconstructing with the same config reproduces identical base shapes, and
        # load_state_dict preserves infshapes.
        if self._is_mup_model():
            mup_finalize(self.model)
            LOGGER.info("Using μP (base shapes set in-backbone; external params finalized)")

        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        LOGGER.info(
            f"Instantiated model {type(self.model.net).__name__} "
            f"with {num_parameters} learnable parameters"
        )
        def list_all_layer_names(model: nn.Module):
            """
            List all named modules in the model.
            Useful for finding the right layer to hook.
            """
            LOGGER.info("All named modules:")
            for name, module in model.named_modules():
                if name:  # Skip empty names
                    LOGGER.info(f"  {name}: {module.__class__.__name__}")
        list_all_layer_names(self.model)

        # === EMA setup ===
        if self.cfg.ema:
            LOGGER.info("Using EMA for validation and eval")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            )
        else:
            self.ema = None
            LOGGER.info("Not using EMA")

        # === Warm start ===
        if self.warm_start:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = _torch_load(model_path, map_location=self.device, weights_only=False)
                LOGGER.info(f"Loading model from {model_path}")
                self.model.load_state_dict(state_dict["model"])

                if self.ema is not None and "ema" in state_dict:
                    self.ema.load_state_dict(state_dict["ema"])
                    LOGGER.info(f"Loaded EMA from {model_path}")
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load model from {model_path}, training model from scratch"
                )

        # === Move to device ===
        if torch.cuda.device_count() > 1:
            LOGGER.info("Using", torch.cuda.device_count(), "GPUs")
            dist.init_process_group("nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.model = DDP(self.model.to(torch.device("cuda")), device_ids=[int(os.environ["LOCAL_RANK"])])
        self.model.to(self.device, dtype=self.dtype)
        if self.ema is not None:
            self.ema.to(self.device)

    def _init(self):
        run_name = self._init_experiment()
        if self.is_main_process():
            self._init_directory()

        if self.cfg.use_mlflow:
            experiment_id = self._init_mlflow()
        else:
            experiment_id = None

        # initialize environment
        self._init_logger()
        self._init_backend()

        return experiment_id, run_name

    def _init_experiment(self):
        self.warm_start = False if self.cfg.warm_start_idx is None else True

        if not self.warm_start:
            if self.cfg.run_dir is not None:
                # Explicit run_dir override (e.g. from run_trial.py for DyHPO sweeps).
                # Use it directly; derive run_name from the last path component for logging.
                run_dir  = self.cfg.run_dir
                run_name = os.path.basename(run_dir)
                if self.cfg.run_name is None:
                    pass  # run_name already set above
                else:
                    run_name = self.cfg.run_name
            else:
                if self.cfg.run_name is None:
                    modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
                    now = datetime.datetime.now()
                    rnd_number = np.random.randint(low=0, high=9999)
                    run_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{modelname}_{rnd_number:04}"
                    self.cfg.run_name = run_name
                else:
                    run_name = self.cfg.run_name

                run_dir = os.path.join(
                    self.cfg.base_dir, "runs", self.cfg.exp_name, run_name
                )
            run_idx = 0
            LOGGER.info(f"Creating new experiment {self.cfg.exp_name}/{run_name}")

        else:
            run_name = self.cfg.run_name
            run_idx = self.cfg.run_idx + 1
            LOGGER.info(
                f"Warm-starting from existing experiment {self.cfg.exp_name}/{run_name} for run {run_idx}"
            )

        with open_dict(self.cfg):
            self.cfg.run_idx = run_idx
            if not self.warm_start:
                self.cfg.warm_start_idx = 0
                self.cfg.run_name = run_name
                self.cfg.run_dir = run_dir

            # only use mlflow if save=True
            self.cfg.use_mlflow = (
                False if self.cfg.save == False else self.cfg.use_mlflow
            )

        # set seed
        if self.cfg.seed is not None:
            LOGGER.info(f"Using seed {self.cfg.seed}")
            torch.random.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        return run_name

    def _init_mlflow(self):
        # mlflow tracking location
        # mlflow.start_run(if you )
        Path(self.cfg.mlflow.db).parent.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{Path(self.cfg.mlflow.db).parent.resolve()}")
        # mlflow.set_tracking_uri(f"sqlite:///{Path(self.cfg.mlflow.db).resolve()}")

        Path(self.cfg.mlflow.artifacts).mkdir(exist_ok=True)
        try:
            # artifacts not supported
            # mlflow call triggers alembic.runtime.migration logger to shout -> shut it down
            logging.disable(logging.WARNING)
            experiment_id = mlflow.create_experiment(
                self.cfg.exp_name,
                artifact_location=f"file:{Path(self.cfg.mlflow.artifacts)}",
            )
            logging.disable(logging.DEBUG)
            LOGGER.info(
                f"Created mlflow experiment {self.cfg.exp_name} with id {experiment_id}"
            )
        except mlflow.exceptions.MlflowException:
            LOGGER.info(f"Using existing mlflow experiment {self.cfg.exp_name}")
            logging.disable(logging.DEBUG)

        experiment = mlflow.set_experiment(self.cfg.exp_name)
        experiment_id = experiment.experiment_id

        # Check if the meta.yaml file exists
        meta_yaml_path = Path(self.cfg.mlflow.db).parent / experiment_id / 'meta.yaml'
        if not meta_yaml_path.exists():
            LOGGER.error(f"meta.yaml file does not exist at {meta_yaml_path}")
            raise mlflow.exceptions.MissingConfigException(f"Yaml file '{meta_yaml_path}' does not exist.")


        LOGGER.info(f"Set experiment {self.cfg.exp_name} with id {experiment_id}")
        return experiment_id

    def _init_directory(self):
        if not self.cfg.save:
            LOGGER.info(f"Running with save=False, i.e. no outputs will be saved")
            return

        # create experiment directory
        run_dir = Path(os.path.abspath(self.cfg.run_dir))
        if run_dir.exists() and not self.warm_start:
            raise ValueError(
                f"Experiment in directory {self.cfg.run_dir} alredy exists. Aborting."
            )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

        # save source
        if self.cfg.save_source:
            zip_name = os.path.join(self.cfg.run_dir, "source.zip")
            LOGGER.debug(f"Saving source to {zip_name}")
            zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
            path_experiment = os.path.join(self.cfg.base_dir, "experiments")
            for path in [path_experiment]:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, path))
            zipf.close()

    def _init_logger(self):
        # silence other loggers
        # (every app has a logger, eg hydra, torch, mlflow, matplotlib, fontTools...)
        # for name, other_logger in logging.root.manager.loggerDict.items():
        #     if not "lorentz-gatr" in name:
        #         other_logger.level = logging.WARNING

        if logger.LOGGING_INITIALIZED:
            LOGGER.info("Logger already initialized")
            return

        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)

        # init file_handler
        if self.cfg.save:
            file_handler = logging.FileHandler(
                Path(self.cfg.run_dir) / f"out_{self.cfg.run_idx}.log"
            )
            file_handler.setFormatter(FORMATTER)
            file_handler.setLevel(logging.DEBUG)
            LOGGER.addHandler(file_handler)

        # init stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOGGER.level)
        stream_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(stream_handler)

        # flush memory to stream_handler
        # this allows to catch logs that were created before the logger was initialized
        MEMORY_HANDLER.setTarget(
            stream_handler
        )  # can only flush to one handler, choose stream_handler
        MEMORY_HANDLER.close()
        LOGGER.removeHandler(MEMORY_HANDLER)

        # add new handlers to logger
        LOGGER.propagate = False  # avoid duplicate log outputs

        logger.LOGGING_INITIALIZED = True
        LOGGER.debug("Logger initialized")

    def _init_backend(self):
        self.device = get_device()
        LOGGER.info(f"Using device {self.device}")

        if (
            self.cfg.training.float16
            and self.device == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            self.dtype = torch.bfloat16
            LOGGER.debug("Using dtype bfloat16")
        elif self.cfg.training.float16:
            self.dtype = torch.float16
            LOGGER.debug(
                "Using dtype float16 (bfloat16 is not supported by environment)"
            )
        else:
            self.dtype = torch.float32
            LOGGER.debug("Using dtype float32")

        # TF32: ~2x matmul on A100 (Ampere+) at ~1e-3 relative precision; a no-op on
        # V100 (Volta has no TF32). Only the bulk fp32 matmuls are affected; the
        # frame math stays fp32 (pinned by minimum_autocast_precision). A/B the loss
        # on A100 before trusting it for precision-sensitive amplitude fits.
        if self.cfg.training.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.backends.cuda.enable_flash_sdp(self.cfg.training.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.training.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(
            self.cfg.training.enable_mem_efficient_sdp
        )
        if self.cfg.training.force_xformers:
            LOGGER.debug("Forcing use of xformers' attention implementation")
            gatr.primitives.attention.FORCE_XFORMERS = True


    def _init_optimizer(self, param_groups=None):
        if param_groups is None:
            ft = self.cfg.get("fine_tune", None)
            if (
                ft is not None
                and ft.get("pretrained_path", None) is not None
                and (ft.get("layer_decay", 1.0) != 1.0 or ft.get("freeze_blocks", []))
            ):
                from fine_tune import build_ft_param_groups
                param_groups = build_ft_param_groups(
                    self.model,
                    base_lr=self.cfg.training.lr,
                    lr_scale=ft.get("lr_scale", 1.0),
                    layer_decay=ft.get("layer_decay", 1.0),
                )
            elif ft is not None and ft.get("pretrained_path", None) is not None:
                # Fine-tuning without LLRD: global lr_scale only
                lr = self.cfg.training.lr * ft.get("lr_scale", 1.0)
                param_groups = [{"params": list(self.model.parameters()), "lr": lr}]
            else:
                param_groups = [
                    {"params": self.model.parameters(), "lr": self.cfg.training.lr}
                ]

        # Check if we should use μP optimizers
        use_mup = self._is_mup_model()

        # fused=True fuses the Adam/AdamW step into a single multi-tensor CUDA kernel
        # (params already on CUDA here — init_model moved them). MuAdam/MuAdamW pass
        # **kwargs straight through to the underlying torch optimizer, so this reaches
        # the μP path too. CUDA-only; ignored for other optimizers.
        adam_extra = {}
        if (
            self.cfg.training.get("fused_optimizer", True)
            and torch.cuda.is_available()
            and self.cfg.training.optimizer in ("Adam", "AdamW")
        ):
            adam_extra["fused"] = True

        if self.cfg.training.optimizer == "Adam":
            if use_mup and self.cfg.training.mup_use_optim:
                self.optimizer = MuAdam(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                    **adam_extra,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                    **adam_extra,
                )

        elif self.cfg.training.optimizer == "AdamW":
            if use_mup and self.cfg.training.mup_use_optim:
                self.optimizer = MuAdamW(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                    **adam_extra,
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                    **adam_extra,
                )

        elif self.cfg.training.optimizer == "RAdam":
            # no MuRAdam implemented yet
            self.optimizer = torch.optim.RAdam(
                param_groups,
                betas=self.cfg.training.betas,
                eps=self.cfg.training.eps,
                weight_decay=self.cfg.training.weight_decay,
            )

        elif self.cfg.training.optimizer == "Lion":
            # Lion is not μP-aware (you'd need to implement a MuLion if required)
            self.optimizer = Lion(
                param_groups,
                betas=self.cfg.training.betas,
                weight_decay=self.cfg.training.weight_decay,
            )

        elif self.cfg.training.optimizer == "ScheduleFree":
            # also not μP-aware
            self.optimizer = schedulefree.AdamWScheduleFree(
                param_groups,
                betas=self.cfg.training.betas,
                weight_decay=self.cfg.training.weight_decay,
            )

        else:
            raise ValueError(f"Optimizer {self.cfg.training.optimizer} not implemented")

        LOGGER.debug(
            f"Using optimizer {self.cfg.training.optimizer}{' (μP)' if use_mup else ''} with lr={self.cfg.training.lr}"
        )

        # load existing optimizer if specified
        if self.warm_start:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = _torch_load(model_path, map_location=self.device, weights_only=False)["optimizer"]
                LOGGER.info(f"Loading optimizer from {model_path}")
                self.optimizer.load_state_dict(state_dict)
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load optimizer from {model_path}, starting from scratch"
                )


    def _init_scheduler(self):
        if self.cfg.training.scheduler is None:
            self.scheduler = None  # constant lr
            LOGGER.info("Using no scheduler")
        elif self.cfg.training.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.training.lr * self.cfg.training.onecycle_max_lr,
                pct_start=self.cfg.training.onecycle_pct_start,
                total_steps=int(
                    self.cfg.training.iterations * self.cfg.training.scheduler_scale
                ),
            )
            LOGGER.info('Using OneCycleLR scheduler')
        elif self.cfg.training.scheduler == "CosineAnnealingLR":
            is_dyhpo_run = self.cfg.training.get("is_dyhpo_run", False)
            # increment_steps is the number of steps for this fidelity increment.
            # For DyHPO cold starts it equals training.iterations; for warm starts
            # it is passed explicitly via training.increment_steps.
            increment_steps = self.cfg.training.get("increment_steps", None) or self.cfg.training.iterations

            warmup_frac = self.cfg.training.get("cosanneal_warmup_frac", 0.0)
            if warmup_frac > 0:
                # Fraction-based: scale warmup to the current increment so it is
                # meaningful at every fidelity level (short or long run).
                warmup = int(warmup_frac * increment_steps)
            elif is_dyhpo_run:
                # Fallback for DyHPO runs that still use absolute warmup_steps:
                # cap at 5 % of the increment to prevent warmup > run length on
                # short fidelity levels or LR shock on warm-start resumes.
                cfg_warmup = self.cfg.training.cosanneal_warmup_steps
                warmup = min(cfg_warmup, int(0.05 * increment_steps))
            else:
                warmup = self.cfg.training.cosanneal_warmup_steps

            self.scheduler = cosine_warmup_scheduler(
                self.optimizer,
                warmup,
                T_max=int(increment_steps * self.cfg.training.scheduler_scale),
                eta_min=self.cfg.training.cosanneal_eta_min,
            )
            if is_dyhpo_run:
                run_kind = "resume" if self.warm_start else "cold start"
                LOGGER.info(
                    f'Using CosineAnnealingLR scheduler'
                    f' (DyHPO {run_kind}, increment={increment_steps} steps, warmup={warmup})'
                )
            else:
                LOGGER.info('Using CosineAnnealingLR scheduler')
        elif self.cfg.training.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.training.reduceplateau_factor,
                patience=self.cfg.training.reduceplateau_patience,
            )
            LOGGER.info('Using ReduceLROnPlateau scheduler')
        else:
            raise ValueError(
                f"Learning rate scheduler {self.cfg.training.scheduler} not implemented"
            )

        LOGGER.debug(f"Using learning rate scheduler {self.cfg.training.scheduler}")

        # load existing scheduler if specified — skip for DyHPO resume (fresh cycle per increment)
        is_dyhpo_resume = self.cfg.training.get("is_dyhpo_run", False) and self.warm_start
        if self.warm_start and self.scheduler is not None and not is_dyhpo_resume:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = _torch_load(model_path, map_location="cpu")["scheduler"]
                LOGGER.info(f"Loading scheduler from {model_path}")
                self.scheduler.load_state_dict(state_dict)
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load scheduler from {model_path}, starting from scratch"
                )

    def train(self):
        self.train_lr, self.train_loss, self.val_loss = [], [], []
        self.train_grad_norm, self.train_loss_no_reg, self.train_mse = [], [], []
        self.val_loss_no_reg, self.val_mse = [], []
        self.proc_val_losses        = {}   # populated by AmplitudeExperiment._validate
        self.proc_val_losses_no_reg = {}
        self.train_metrics = self._init_metrics()
        self.val_metrics   = self._init_metrics()

        smallest_val_loss, smallest_val_loss_step = 1e10, 0
        smallest_val_loss_no_reg = 1e10   # checkpoint selection uses this when available
        patience  = 0
        results   = []
        iterator  = iter(self._cycle(self.train_loader))
        avg_iter_time = 0

        if self.cfg.find_BS:
            gradients, S_estimates, G2_estimates, B_simples = [], [], [], []

        # Log-spaced checkpoints at which to run _evals (save preds / intrinsic dim)
        oom          = math.floor(np.log10(self.cfg.training.iterations))
        mantissa     = self.cfg.training.iterations / (10 ** oom)
        iters_to_eval = set(math.floor(mantissa * 10 ** (i / 2)) for i in range(0, 2 * oom + 1))

        # validate_frac: if > 0, sets validate_every_n_steps = frac * effective_steps,
        # giving a fixed number of validation points regardless of total iterations.
        # Uses increment_steps in DyHPO mode so each fidelity level also gets ~1/frac validations.
        validate_frac = self.cfg.training.get("validate_frac", 0.0)
        if validate_frac > 0:
            inc = self.cfg.training.get("increment_steps", None)
            effective_steps = inc if inc is not None else self.cfg.training.iterations
            with open_dict(self.cfg):
                self.cfg.training.validate_every_n_steps = max(1, int(validate_frac * effective_steps))

        LOGGER.info(
            f"Starting to train for {self.cfg.training.iterations} iterations "
            f"= {self.cfg.training.iterations / len(self.train_loader):.1f} epochs "
            f"on a dataset with {len(self.train_loader)} batches | "
            f"early stopping patience {self.cfg.training.es_patience} | "
            f"validating every {self.cfg.training.validate_every_n_steps} steps"
            + (f" (validate_frac={validate_frac})" if validate_frac > 0 else "")
        )
        self.training_start_time = time.time()

        # --- optional per-iter profiling: split dataloading vs compute ---
        # LLOCA_PROFILE_STEP=1 attributes each iteration to next(iterator)
        # (collate + H2D — serial with the GPU when num_workers=0) vs _step
        # (forward/backward/opt). CUDA syncs bracket each segment so the split is
        # accurate; this perturbs absolute throughput, so use it only as a
        # diagnostic run, not for the headline avg-iter number.
        profile_step = os.environ.get("LLOCA_PROFILE_STEP", "0") == "1"
        prof_data_sum = prof_step_sum = 0.0
        prof_n = 0
        _on_cuda = self.device == torch.device("cuda")

        # --- evaluate at initialisation (step 0, before any gradient update) ---
        if self.is_main_process() and self.cfg.training.save_intermediate:
            self._evals(results, smallest_val_loss_step, step=0)
            if self.cfg.training.save_gradients:
                self._save_gradients(step=0)

        for step in range(self.cfg.training.iterations):
            iter_time_start = time.time()

            # --- training step ---
            self.model.train()
            if self.cfg.training.optimizer == "ScheduleFree":
                self.optimizer.train()
            if profile_step:
                if _on_cuda:
                    torch.cuda.synchronize()
                _t0 = time.time()
                data = next(iterator)
                if _on_cuda:
                    torch.cuda.synchronize()
                _t1 = time.time()
                self._step(data, step)
                if _on_cuda:
                    torch.cuda.synchronize()
                _t2 = time.time()
                # skip the first few steps (lazy init, cudnn autotune, worker warmup)
                if step >= 5:
                    prof_data_sum += _t1 - _t0
                    prof_step_sum += _t2 - _t1
                    prof_n += 1
            else:
                data = next(iterator)
                self._step(data, step)

            # --- batch-size finder ---
            if self.cfg.find_BS:
                grads = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
                gradients.append(grads.detach().clone())
                if len(gradients) > self.cfg.BS_finding.n_batches_big:
                    gradients = gradients[-self.cfg.BS_finding.n_batches_big:]
                    grad_big   = torch.stack(gradients).mean(dim=0)
                    grad_small = torch.stack(random.sample(gradients, self.cfg.BS_finding.n_batches_small)).mean(dim=0)
                    B_big   = self.cfg.training.batchsize * self.cfg.BS_finding.n_batches_big
                    B_small = self.cfg.training.batchsize * self.cfg.BS_finding.n_batches_small
                    S_estimate  = abs(1 / (1/B_small - 1/B_big) * (grad_small.norm()**2 - grad_big.norm()**2))
                    G2_estimate = abs(1 / (B_big - B_small) * (B_big * grad_big.norm()**2 - B_small * grad_small.norm()**2))
                    S_estimates.append(S_estimate.item())
                    G2_estimates.append(G2_estimate.item())
                    ema_S  = pd.Series(S_estimates).ewm(alpha=self.cfg.BS_finding.ema_alpha,  adjust=False).mean()
                    ema_G2 = pd.Series(G2_estimates).ewm(alpha=self.cfg.BS_finding.ema_alpha, adjust=False).mean()
                    B_simple = ema_S.iloc[-1] / ema_G2.iloc[-1]
                    B_simples.append(B_simple.item())
                    if step % 100 == 0:
                        LOGGER.info(f"S={S_estimate.item():.3e}  G2={G2_estimate.item():.3e}  B_simple={B_simple.item():.3e}")

            avg_iter_time = (avg_iter_time * step + (time.time() - iter_time_start)) / (step + 1)

            # --- validation ---
            is_val_step = (step + 1) % self.cfg.training.validate_every_n_steps == 0
            is_last_step = (step + 1) == self.cfg.training.iterations

            if is_val_step or is_last_step:
                val_loss = self._validate(step)
                # Select checkpoint on no-reg loss when available; fall back to
                # regularized loss only if no-reg was never computed.
                val_loss_no_reg_now = self.val_loss_no_reg[-1] if self.val_loss_no_reg else None
                selection_loss = val_loss_no_reg_now if val_loss_no_reg_now is not None else val_loss
                improved = selection_loss < smallest_val_loss

                if improved:
                    smallest_val_loss        = selection_loss
                    smallest_val_loss_step   = step
                    smallest_val_loss_no_reg = val_loss_no_reg_now if val_loss_no_reg_now is not None else val_loss
                    patience = 0
                    if self.cfg.training.es_load_best_model:
                        self._save_model(step, f"model_run{self.cfg.run_idx}_best.pt")
                else:
                    patience += self.cfg.training.validate_every_n_steps
                    if patience > self.cfg.training.es_patience:
                        LOGGER.info(f"Early stopping at iteration {step + 1} = epoch {(step + 1) / len(self.train_loader):.1f}")
                        break

                if self.cfg.training.scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(selection_loss)

                # --- _evals immediately after validation, always on val steps ---
                # This ensures _evals always reflects the current best model
                # (smallest_val_loss_step is up-to-date), not a stale snapshot.
                if self.is_main_process() and self.cfg.training.save_intermediate:
                    self._evals(results, smallest_val_loss_step, step)
                    if self.cfg.training.save_gradients and (step + 1) in iters_to_eval:
                        self._save_gradients(step)
                    if self.cfg.training.save_weights and (step + 1) in iters_to_eval:
                        self._save_weights(step)

            # --- progress logging ---
            dt = time.time() - self.training_start_time
            log_steps = {0, 999} | set(round(x) for x in np.linspace(0, self.cfg.training.iterations - 1, num=10))
            if step in log_steps:
                dt_est = dt * self.cfg.training.iterations / (step + 1)
                vram_str = ""
                if self.device == torch.device("cuda"):
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    peak_mb  = torch.cuda.max_memory_allocated() / 1024**2
                    vram_str = f" | VRAM alloc: {alloc_mb:.0f} MB, peak: {peak_mb:.0f} MB"
                prof_str = ""
                if profile_step and prof_n > 0:
                    d_ms = prof_data_sum / prof_n * 1000
                    s_ms = prof_step_sum / prof_n * 1000
                    tot  = d_ms + s_ms
                    prof_str = (
                        f" | PROFILE n={prof_n}: data {d_ms:.1f}ms ({100*d_ms/tot:.0f}%), "
                        f"step {s_ms:.1f}ms ({100*s_ms/tot:.0f}%)"
                    )
                LOGGER.info(
                    f"Finished iteration {step + 1} after {dt:.2f}s | "
                    f"estimate: {dt_est/60:.2f}min = {dt_est/3600:.2f}h"
                    + vram_str + prof_str
                )

        # --- end of training ---
        dt = time.time() - self.training_start_time
        if profile_step and prof_n > 0:
            d_ms = prof_data_sum / prof_n * 1000
            s_ms = prof_step_sum / prof_n * 1000
            tot  = d_ms + s_ms
            LOGGER.info(
                f"PROFILE summary over {prof_n} iters: "
                f"dataloading {d_ms:.1f}ms/iter ({100*d_ms/tot:.0f}%), "
                f"compute(_step) {s_ms:.1f}ms/iter ({100*s_ms/tot:.0f}%), "
                f"total {tot:.1f}ms/iter"
            )
        LOGGER.info(
            f"Finished training: {step + 1} iterations = {(step + 1) / len(self.train_loader):.1f} epochs "
            f"in {dt/60:.2f}min (avg {avg_iter_time:.4f}s/iter)"
        )
        if self.cfg.use_mlflow:
            log_mlflow("iterations", step + 1)
            log_mlflow("epochs",     (step + 1) / len(self.train_loader))
            log_mlflow("traintime",  dt / 3600)

        # Load best model at end of training
        if self.cfg.training.es_load_best_model:
            self._load_previous_best_model()

        # Write best val loss for external HPO tools (e.g. Optuna sweep).
        # Reports the no-regularization loss so the surrogate isn't biased by the
        # regularization strength (which is itself a tuned hyperparameter).
        # Falls back to the regularized loss if no-reg tracking wasn't enabled.
        result_path = self.cfg.training.get("result_path", None)
        if result_path:
            import json
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            best_loss = smallest_val_loss_no_reg if smallest_val_loss_no_reg < 1e10 else smallest_val_loss
            result = {"val_loss": float(best_loss), "traintime_hours": dt / 3600.0}
            result.update(self._result_extra())
            with open(result_path, "w") as _f:
                json.dump(result, _f)

    def _result_extra(self) -> dict:
        """Subclasses can override to add extra fields to the result JSON."""
        return {}

        if self.cfg.find_BS:
            self._plot_bs_finding(S_estimates, G2_estimates, B_simples, ema_S, ema_G2)

    @staticmethod
    def _cycle(iterable):
        while True:
            for x in iterable:
                yield x
        
    def _plot_bs_finding(self, S_estimates, G2_estimates, B_simples, ema_S, ema_G2):
        for values, ema, label, color, ema_color, fname in [
            (S_estimates,  ema_S,  "S",        "blue",  "cyan",  "S_estimation"),
            (G2_estimates, ema_G2, "G2",       "green", "lime",  "G2_estimation"),
            (B_simples,    None,   "B_simple", "red",   None,    "B_simple_estimation"),
        ]:
            plt.figure()
            plt.plot(range(len(values)), values, label=f"{label} estimates", color=color)
            if ema is not None:
                plt.plot(range(len(ema)), ema, label=f"EMA of {label}", color=ema_color)
            plt.xlabel("Iteration")
            plt.ylabel(f"Estimated {label}")
            plt.title(f"Estimation of {label} over Iterations")
            plt.yscale("log")
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(self.cfg.run_dir, f"{fname}_run{self.cfg.run_idx}.pdf"))
            plt.close()

    def _evals(self, results, smallest_val_loss_step, step):
        if not self.is_main_process():
            return
        LOGGER.info(
            f"### Evaluating model (best from iteration {smallest_val_loss_step + 1}) "
            f"at iteration {step + 1} ###"
        )
        self._save_model(step, f"model_run{self.cfg.run_idx}_current.pt")

        if self.cfg.training.load_best_previous:
            try:
                self._load_previous_best_model()
            except FileNotFoundError:
                self._load_previous_current_model()

        self.evaluate()

        if self.cfg.training.get_ID:
            id_time = time.time()
            ID_mean, ID_std = get_intrinsic_dim(
                model=self.model,
                input_dataloader=self.test_loader,
                nsamples=min(1e3, self.cfg.data.subsample * self.cfg.training.train_test_val[1]),
                bs=self.cfg.training.batchsize,
                divs=2,
                res=3,
                call_model_fn=self.call_model_fn,
            )
            LOGGER.info(f"Intrinsic Dimension estimation took {time.time() - id_time:.2f}s")
        else:
            ID_mean = ID_std = None

        for dataset, dataset_results in self.results.items():
            for split, split_results in dataset_results.items():
                for processing_type, metrics in split_results.items():
                    if self.cfg.training.save_preds_intermediate or (step + 1) == self.cfg.training.iterations:
                        self._save_preds_intermediate(step + 1, dataset, split, processing_type, metrics)
                    results.append({
                        "run_id":          self.cfg.run_idx,
                        "samplesize":      int(self.cfg.data.subsample),
                        "dataset":         dataset,
                        "split":           split,
                        "processing_type": processing_type,
                        "mse":             metrics.get("mse"),
                        "l1":              metrics.get("l1"),
                        "l1_rel":          metrics.get("l1_rel"),
                        "iter":            step + 1,
                        "iter_evaluated":  smallest_val_loss_step + 1,
                        "ID":              [ID_mean, ID_std] if self.cfg.training.get_ID else None,
                        "time":            time.time() - self.training_start_time,
                    })

        # Restore the current (non-best) weights so training can continue
        self._load_previous_current_model()

        pkl_path = os.path.join(self.cfg.run_dir, "results_intermediate.pkl")
        LOGGER.info(f"Saving results to {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)


    def _save_gradients(self, step):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Plotting gradients to {plot_path}/gradients_{step+1}.pdf")
        plot_gradients(file=plot_path,model=self.model, iteration=step+1)

    def _save_weights(self, step):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Plotting weights to {plot_path}/weights_{step+1}.pdf")
        plot_weights(file=plot_path,model=self.model, iteration=step+1)

    def _save_preds_intermediate(self, step, dataset, split, processing_type, metrics):
        os.makedirs(os.path.join(self.cfg.run_dir, "preds"), exist_ok=True)
        pred_path = os.path.join(self.cfg.run_dir, f"preds/{step}_{dataset}_{split}_{processing_type}_pred.npy")
        np.save(pred_path, metrics["prediction"])
        if self.cfg.training.loss == "HETEROSC" and processing_type == "preprocessed":
            sigma_path = os.path.join(self.cfg.run_dir, f"preds/{step}_{dataset}_{split}_{processing_type}_sigmas.npy")
            np.save(sigma_path, metrics["sigmas"])
            pull_path = os.path.join(self.cfg.run_dir, f"preds/{step}_{dataset}_{split}_{processing_type}_pull.npy")
            np.save(pull_path, metrics["pull"])

    def _is_mup_model(self):
        return self.cfg.model.net._target_ in (
            "models.mup_mlp.MuMLP",
            "models.lloca.LLOCAMuPTransformer",
            "models.lgatr_slim_mup.MuPLGATrSlim",
            "models.lgatr_mup.MuPLGATr",
        )

    def _load_pretrained_weights(self, pretrained_path: str, reset_output_head: bool = False):
        """Load model weights from an external pretrained checkpoint.

        Unlike warm-start (which resumes the same run), this loads only model
        weights — optimizer and scheduler state are always started fresh.
        """
        try:
            state_dict = _torch_load(pretrained_path, map_location=self.device, weights_only=False)["model"]
            LOGGER.info(f"Fine-tuning: loading pretrained weights from {pretrained_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        if reset_output_head:
            head = self.model.net.net.linear_out
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            LOGGER.info("Fine-tuning: reset output head (linear_out) to zeros.")

        # No μP bookkeeping needed: base shapes were set when the backbone was
        # constructed and load_state_dict preserves the per-parameter infshapes.

    def _init_ewc(self):
        self.ewc = None

    def _load_model_weights(self, model_path):
        try:
            state_dict = _torch_load(model_path, map_location=self.device, weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(f"Cannot load model from {model_path}")
        # base shapes are intrinsic to the constructed backbone; load_state_dict
        # preserves infshapes, so no set_base_shapes call is needed here.

    def _load_previous_model(self, step):
        self._load_model_weights(os.path.join(
            self.cfg.run_dir, "models", f"model_run{self.cfg.run_idx}_it{step}.pt"
        ))

    def _load_previous_best_model(self):
        self._load_model_weights(os.path.join(
            self.cfg.run_dir, "models", f"model_run{self.cfg.run_idx}_best.pt"
        ))

    def _load_previous_current_model(self):
        self._load_model_weights(os.path.join(
            self.cfg.run_dir, "models", f"model_run{self.cfg.run_idx}_current.pt"
        ))
            
    def _step(self, data, step):
        # actual update step
        loss, loss_no_reg, mse_val = self._batch_loss(data)
        if self.ewc is not None:
            ewc_lambda = self.cfg.fine_tune.ewc.get("lambda", 1000.0)
            loss = loss + (ewc_lambda / 2.0) * self.ewc.penalty(self.model)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.training.clip_grad_value is not None:
            # clip gradients at a certain value (this is dangerous!)
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.cfg.training.clip_grad_value,
            )
        # rescale gradients such that their norm matches a given number
        grad_norm_t = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.training.clip_grad_norm,
            error_if_nonfinite=False,
        )

        # --- materialise the per-step scalars (the one unavoidable D2H sync) ---
        # train_grad_norm is always logged, so grad_norm must be read every step;
        # we fold loss (and the no-reg loss, if plotted) into the SAME .tolist() so
        # the whole step costs one sync instead of 3-4. The old per-step syncs we
        # removed: loss.item() before backward (in _batch_loss), the isfinite
        # assert, clip_grad_norm_().cpu().item(), and a duplicate loss.item().
        # LLOCA_SYNC=blocking reproduces the original separate syncs for A/B timing.
        sync_blocking = os.environ.get("LLOCA_SYNC", "deferred") == "blocking"
        want_lnr = self.cfg.plotting.get("plot_without_regularization", False)

        if sync_blocking:
            grad_norm = grad_norm_t.cpu().item()
            loss_val  = None   # original read loss.item() after optimizer.step()
            lnr_val   = loss_no_reg.item() if want_lnr else None
        else:
            # .float() so the stack is dtype-uniform (loss may be fp16/bf16 while the
            # grad norm is fp32); all are 0-dim so the cast is free.
            scalars = [loss.detach().float(), grad_norm_t.float()]
            if want_lnr:
                scalars.append(loss_no_reg.float())
            vals     = torch.stack(scalars).tolist()      # single fused GPU→CPU sync
            loss_val = vals[0]
            grad_norm = vals[1]
            lnr_val  = vals[2] if want_lnr else None
            # NaN/inf guard: replaces the per-step isfinite assert (which synced).
            # A non-finite loss yields a non-finite grad_norm; skip the update so a
            # transient blow-up doesn't corrupt the weights (better than crashing).
            if not math.isfinite(grad_norm):
                LOGGER.warning(f"Skipping update, non-finite gradient norm {grad_norm}")
                return

        if step > MIN_STEP_SKIP and self.cfg.training.max_grad_norm is not None:
            if grad_norm > self.cfg.training.max_grad_norm:
                LOGGER.warning(
                    f"Skipping update, gradient norm {grad_norm} exceeds maximum {self.cfg.training.max_grad_norm}"
                )
                return

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update()

        if self.cfg.training.scheduler in ["OneCycleLR", "CosineAnnealingLR"]:
            self.scheduler.step()

        # collect metrics
        if loss_val is None:
            loss_val = loss.item()   # blocking path: read post-step (matches original)
        self.train_loss.append(loss_val)
        self.train_lr.append(self.optimizer.param_groups[0]["lr"])
        self.train_grad_norm.append(grad_norm)
        if want_lnr:
            self.train_loss_no_reg.append(lnr_val)
        if mse_val is not None and self.cfg.plotting.get("plot_mse_het", False):
            self.train_mse.append(mse_val)

        # log to mlflow
        if (
            self.cfg.use_mlflow
            and self.cfg.training.log_every_n_steps != 0
            and step % self.cfg.training.log_every_n_steps == 0
        ):
            log_dict = {
                "loss": loss_val,
                "lr": self.train_lr[-1],
                "time_per_step": (time.time() - self.training_start_time) / (step + 1),
                "grad_norm": grad_norm,
            }
            for key, values in log_dict.items():
                log_mlflow(f"train.{key}", values, step=step)

    def _validate(self, step):
        start_time_validate = time.time()
        losses = []
        losses_no_reg = []
        mse_vals = []
        metrics = self._init_metrics()
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            for data in self.val_loader:
                if self.ema is not None:
                    with self.ema.average_parameters():
                        loss, loss_no_reg, mse_val = self._batch_loss(data)
                else:
                    loss, loss_no_reg, mse_val = self._batch_loss(data)
                losses.append(loss.cpu().item())
                if loss_no_reg is not None:
                    losses_no_reg.append(loss_no_reg.item())   # now a detached tensor
                if mse_val is not None:
                    mse_vals.append(mse_val)

        val_loss = np.mean(losses)

        # Release validation tensors from the GPU memory pool to avoid fragmentation
        # that would slow down subsequent training steps.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if ((step + 1) % self.cfg.training.validate_every_n_steps == 0):
            self.val_loss.append(val_loss)
            if losses_no_reg:
                self.val_loss_no_reg.append(np.mean(losses_no_reg))
            if mse_vals:
                self.val_mse.append(np.mean(mse_vals))
            if self.cfg.use_mlflow:
                log_mlflow("val.loss", val_loss, step=step)

        end_time_validate = time.time()
        return val_loss

    def _save_config(self, filename, to_mlflow=False):
        # Save config
        if not self.cfg.save:
            return

        config_filename = Path(self.cfg.run_dir) / filename
        LOGGER.debug(f"Saving config at {config_filename}")
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

        if to_mlflow and self.cfg.use_mlflow:
            for key, value in flatten_dict(self.cfg).items():
                log_mlflow(key, value, kind="param")

    def _save_model(self, step="end", filename=None):
        if not self.cfg.save:
            return

        if filename is None:
            filename = f"model_run{self.cfg.run_idx}.pt"
        model_path = os.path.join(self.cfg.run_dir, "models", filename)
        LOGGER.debug(f"Saving model at {model_path}")
        if self.ema is not None:
            with self.ema.average_parameters():
                model_state = self.model.state_dict()
        else:
            model_state = self.model.state_dict()
        torch.save(
            {
                "model": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "step": step
            },
            model_path,
        )

    def compress_models(self):
        pt_files = glob.glob(os.path.join(self.cfg.run_dir , "models", "*.pt"))

        if not pt_files:
            LOGGER.info("No .pt files found.")
        else:
            for path in pt_files:
                gz_path = path + ".gz"

                if os.path.exists(gz_path):
                    LOGGER.info(f"Skipping {path} (already compressed)")
                    continue

                LOGGER.info(f"Compressing {path} -> {gz_path}")
                with open(path, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(path)  # Remove original
                LOGGER.info(f"  Done, original removed.")
                
    def count_flops(self, dataloader):
        with FlopCounterMode(self.model) as fcm:
            for data in dataloader:
                for idataset, data_onedataset in enumerate(data):
                    x, y = data_onedataset
                    if self.modelname == "LLOCAMuPTransformer":
                        outputs = self.model(
                            x.to(self.device),
                            type_token=torch.tensor(
                                [self.type_token[idataset]],
                                dtype=torch.long,
                                device=self.device,
                            ),
                            coord_scale=self.coord_scale,
                            mean=self.mom_mean,
                            std=self.mom_std,
                        )
                    else:
                        self.model(
                        x.to(self.device),
                        type_token=torch.tensor(
                            [self.type_token[idataset]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        global_token=torch.tensor(
                            [idataset], dtype=torch.long, device=self.device
                        ),
                    )
        flops = fcm.get_total_flops()
        LOGGER.info(f"FLOPs per forward pass: {flops}")

    def init_physics(self):
        raise NotImplementedError()

    def init_data(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def _init_dataloader(self):
        raise NotImplementedError()

    def _init_loss(self):
        raise NotImplementedError()
    
    def _init_regularization(self):
        raise NotImplementedError()

    def _batch_loss(self, data):
        raise NotImplementedError()

    def _init_metrics(self):
        raise NotImplementedError()