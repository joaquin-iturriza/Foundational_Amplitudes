import numpy as np
import torch
import math
import datetime
import pickle
import os, time
import zipfile
import logging
import pandas as pd
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


cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)

#print("Registered configs:", cs.repo)

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

        if self.cfg.train:
            self._init_optimizer()
            self._init_scheduler()
            self.train()
            self._save_model()

        if self.cfg.evaluate:
            self.evaluate()

        if self.cfg.plot and self.cfg.save:
            self.plot()

        if self.device == torch.device("cuda"):
            max_used = torch.cuda.max_memory_allocated()
            max_total = torch.cuda.mem_get_info()[1]
            LOGGER.info(
                f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
            )
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
                    f"Mixing mvpseudoscalar into scalar is only possible if ga_settings.use_fully_connected_subgroup=True"
                )
                lgatr.layers.linear.MIX_MVPSEUDOSCALAR_INTO_SCALAR = False
        lgatr.layers.mlp.mlp.USE_GEOMETRIC_PRODUCT = (
            self.cfg.ga_settings.use_geometric_product
        )
        lgatr.layers.mlp.geometric_bilinears.ZERO_BIVECTOR = (
            self.cfg.ga_settings.zero_bivector
        )
    
    def init_model(self):
        # initialize model
        print(self.cfg.model)
        self.model = instantiate(self.cfg.model)
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.cfg.use_mlflow:
            log_mlflow("num_parameters", float(num_parameters), step=0)
        LOGGER.info(
            f"Instantiated model {type(self.model.net).__name__} with {num_parameters} learnable parameters"
        )

        if self.cfg.ema:
            LOGGER.info(f"Using EMA for validation and eval")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            )
        else:
            LOGGER.info(f"Not using EMA")
            self.ema = None

        # load existing model if specified
        if self.warm_start:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
                LOGGER.info(f"Loading model from {model_path}")
                self.model.load_state_dict(state_dict)
                if self.ema is not None:
                    LOGGER.info(f"Loading EMA from {model_path}")
                    state_dict = torch.load(model_path, map_location="cpu")["ema"]
                    self.ema.load_state_dict(state_dict)
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load model from {model_path}, training model from scratch"
                )

        self.model.to(self.device, dtype=self.dtype)
        if self.ema is not None:
            self.ema.to(self.device)

    def _init(self):
        run_name = self._init_experiment()
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
        run_dir = Path(self.cfg.run_dir).resolve()
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
            param_groups = [
                {"params": self.model.parameters(), "lr": self.cfg.training.lr}
            ]

        if self.cfg.training.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                param_groups,
                betas=self.cfg.training.betas,
                eps=self.cfg.training.eps,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif self.cfg.training.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=self.cfg.training.betas,
                eps=self.cfg.training.eps,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif self.cfg.training.optimizer == "RAdam":
            self.optimizer = torch.optim.RAdam(
                param_groups,
                betas=self.cfg.training.betas,
                eps=self.cfg.training.eps,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif self.cfg.training.optimizer == "Lion":
            self.optimizer = Lion(
                param_groups,
                betas=self.cfg.training.betas,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer = schedulefree.AdamWScheduleFree(
                param_groups,
                betas=self.cfg.training.betas,
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.cfg.training.optimizer} not implemented")
        LOGGER.debug(
            f"Using optimizer {self.cfg.training.optimizer} with lr={self.cfg.training.lr}"
        )

        # load existing optimizer if specified
        if self.warm_start:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["optimizer"]
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
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(
                    self.cfg.training.iterations * self.cfg.training.scheduler_scale
                ),
                eta_min=self.cfg.training.cosanneal_eta_min,
            )
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

        # load existing scheduler if specified
        if self.warm_start and self.scheduler is not None:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = torch.load(model_path, map_location="cpu")["scheduler"]
                LOGGER.info(f"Loading scheduler from {model_path}")
                self.scheduler.load_state_dict(state_dict)
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load scheduler from {model_path}, starting from scratch"
                )

    def train(self):
        # performance metrics
        self.train_lr, self.train_loss, self.val_loss, self.train_grad_norm = (
            [],
            [],
            [],
            [],
        )
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()

        # early stopping
        smallest_val_loss, smallest_val_loss_step = 1e10, 0
        patience = 0

        # main train loop
        LOGGER.info(
            f"Starting to train for {self.cfg.training.iterations} iterations "
            f"= {self.cfg.training.iterations / len(self.train_loader):.1f} epochs "
            f"on a dataset with {len(self.train_loader)} batches "
            f"using early stopping with patience {self.cfg.training.es_patience} "
            f"while validating every {self.cfg.training.validate_every_n_steps} iterations"
        )
        self.training_start_time = time.time()

        # recycle trainloader
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        results = []#{}
        #results['samplesize']=int(self.cfg.data.subsample)
        mses_prepd = []
        iters = []
        iters_evaluated = []
        iterator = iter(cycle(self.train_loader))
        for step in range(self.cfg.training.iterations):
            # training
            #iter_time_start = time.time()
            if step > 0:   # skip first step to get evaluation at initialization #### NOT GOOD, FIX LATER
                self.model.train()
                if self.cfg.training.optimizer == "ScheduleFree":
                    self.optimizer.train()
                data = next(iterator)
                self._step(data, step)
            
            if self.cfg.training.save_intermediate or (step+1) == self.cfg.training.iterations: ########## FIX LATER
                oom=math.floor(np.log10(self.cfg.training.iterations))
                mantissa=self.cfg.training.iterations/(10**oom)
                iters_to_eval=[math.floor(mantissa*10**i) for i in range(0,oom+1)]
                if ((step+1) in iters_to_eval) or (step == 0):
                    LOGGER.info(
                            f"### Evaluating in iteration {smallest_val_loss_step+1} ###"
                        )
                    self._save_model(
                        f"model_run{self.cfg.run_idx}_current.pt"
                    )
                    try:
                        self._load_previous_best_model()
                    except FileNotFoundError:
                        self._load_previous_current_model()
                    res = self.evaluate()
                    self._load_previous_current_model()

                    if self.cfg.training.get_ID:  ### ONLY WORKS WITH ONE DATASET, FIX LATER
                        id_time = time.time()
                        ID_mean, ID_std = get_intrinsic_dim(
                            model = self.model,
                            input_dataloader = self.test_loader,
                            nsamples = min(1e4,self.cfg.data.subsample*self.cfg.training.train_test_val[1]),
                            bs=self.cfg.training.batchsize,
                            divs = 5,
                            res = 10,
                            call_model_fn=self.call_model_fn
                        )
                        id_time = time.time() - id_time
                        LOGGER.info(
                            f"Intrinsic Dimension estimation took {id_time:.2f}s"
                        )
                    #### There might be a better way to do this ####
                    # for dataset in res.keys():
                    #     if dataset not in results.keys():
                    #         results[dataset] = {}
                    #     for title in res[dataset].keys():
                    #         if title not in results[dataset].keys():
                    #             results[dataset][title] = {}
                    #         for key in res[dataset][title].keys():
                    #             if key not in results[dataset][title].keys():
                    #                 results[dataset][title][key] = {}
                    #             if "mses" not in results[dataset][title][key]:
                    #                 results[dataset][title][key]["mses"] = []
                    #                 results[dataset][title][key]["l1s"] = []
                    #                 results[dataset][title][key]["l1s_rel"] = []
                    #                 results[dataset][title][key]["iters"] = []
                    #                 results[dataset][title][key]["iters_evaluated"] = []
                    #                 results[dataset][title][key]["time"] = []
                            
                    #             results[dataset][title][key]["mses"].append(res[dataset][title][key]["mse"])
                    #             results[dataset][title][key]["l1s"].append(res[dataset][title][key]["l1"])
                    #             results[dataset][title][key]["l1s_rel"].append(res[dataset][title][key]["l1_rel"])
                    #             results[dataset][title][key]["iters"].append(step+1)
                    #             results[dataset][title][key]["iters_evaluated"].append(smallest_val_loss_step)
                    #             results[dataset][title][key]["time"].append(time.time() - self.training_start_time)                  
                    for dataset, dataset_results in self.results.items():  # Loop through datasets
                        for split, split_results in dataset_results.items():  # Loop through splits (train/val/test)
                            for processing_type, metrics in split_results.items():  # Loop through raw/preprocessed
                                # Create directory for predictions if it doesn't exist
                                if self.cfg.training.save_preds_intermediate or (step+1) == self.cfg.training.iterations:
                                    os.makedirs(os.path.join(self.cfg.run_dir, "preds"), exist_ok=True)
                                    
                                    # Save predictions to file
                                    pred_path = os.path.join(self.cfg.run_dir, f"preds/{step+1}_{dataset}_{split}_{processing_type}_pred.npy")
                                    np.save(pred_path, metrics["prediction"])
                                    
                                    # For heteroscedastic case, also save sigmas if they exist
                                    if self.cfg.training.loss == "HETEROSC" and processing_type == "preprocessed":
                                        sigma_path = os.path.join(self.cfg.run_dir, f"preds/{step+1}_{dataset}_{split}_{processing_type}_sigmas.npy")
                                        np.save(sigma_path, metrics["sigmas"])
                                        pull_path = os.path.join(self.cfg.run_dir, f"preds/{step+1}_{dataset}_{split}_{processing_type}_pull.npy")
                                        np.save(pull_path, metrics["pull"])
                                # Append results to list
                                
                                results.append({
                                    'run_id': self.cfg.run_idx,
                                    'samplesize': int(self.cfg.data.subsample),
                                    'dataset': dataset,
                                    'split': split,  # train/val/test
                                    'processing_type': processing_type,  # raw/preprocessed
                                    'mse': metrics.get('mse'),
                                    'l1': metrics.get('l1'),
                                    'l1_rel': metrics.get('l1_rel'),
                                    'iter': step + 1,
                                    'iter_evaluated': smallest_val_loss_step,
                                    'ID': [ID_mean, ID_std] if self.cfg.training.get_ID else None,
                                    'time': time.time() - self.training_start_time
                                })

            # validation (and early stopping)
            if self.cfg.training.save_gradients:
                if ((step+1) in iters_to_eval): #plot gradients
                    plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
                    os.makedirs(plot_path, exist_ok=True)
                    LOGGER.info(f"Plotting gradients to {plot_path}/gradients_{step+1}.pdf")
                    plot_gradients(file=plot_path,model=self.model, iteration=step+1)
            if self.cfg.training.save_weights:
                if ((step+1) in iters_to_eval): #plot weights
                    plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
                    os.makedirs(plot_path, exist_ok=True)
                    LOGGER.info(f"Plotting weights to {plot_path}/weights_{step+1}.pdf")
                    plot_weights(file=plot_path,model=self.model, iteration=step+1)

            if ((step + 1) % self.cfg.training.validate_every_n_steps == 0) or (step < self.cfg.training.validate_every_n_steps and self.cfg.training.save_intermediate):
                val_loss = self._validate(step)
                if val_loss < smallest_val_loss:
                    #LOGGER.info(
                    #    f"Validation loss improved from {smallest_val_loss:.4f} to {val_loss:.4f} in iteration {step+1}"
                    #)
                    smallest_val_loss = val_loss
                    smallest_val_loss_step = step
                    patience = 0

                    # save best model
                    if self.cfg.training.es_load_best_model:
                        self._save_model(
                            f"model_run{self.cfg.run_idx}_best.pt"#it{smallest_val_loss_step}.pt"
                        )
                else:
                    patience += 1
                    if patience > self.cfg.training.es_patience:
                        LOGGER.info(
                            f"Early stopping in iteration {step} = epoch {step / len(self.train_loader):.1f}"
                        )
                        break  # early stopping

                if self.cfg.training.scheduler in ["ReduceLROnPlateau"]:
                    self.scheduler.step(val_loss)

            # output
            dt = time.time() - self.training_start_time
            if step in [0, 999]:
                dt_estimate = dt * self.cfg.training.iterations / (step + 1)
                LOGGER.info(
                    f"Finished iteration {step+1} after {dt:.2f}s, "
                    f"training time estimate: {dt_estimate/60:.2f}min "
                    f"= {dt_estimate/60**2:.2f}h"
                )

        dt = time.time() - self.training_start_time
        LOGGER.info(
            f"Finished training for {step} iterations = {step / len(self.train_loader):.1f} epochs "
            f"after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )
        if self.cfg.use_mlflow:
            log_mlflow("iterations", step)
            log_mlflow("epochs", step / len(self.train_loader))
            log_mlflow("traintime", dt / 3600)

        # wrap up early stopping
        if self.cfg.training.es_load_best_model:
            self._load_previous_best_model()

        # save results intermediate
        if self.cfg.training.save_intermediate:
            LOGGER.info(
                f"Saving results to {self.cfg.run_dir}/results_intermediate.pkl"
            )
            with open(os.path.join(self.cfg.run_dir, "results_intermediate.pkl"), "wb") as f:
                pickle.dump(results, f)
            # df = pd.DataFrame(results)
            # df.to_pickle(os.path.join(self.cfg.run_dir, "results_intermediate.pkl"))

    def _load_previous_model(self,step):
        model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"model_run{self.cfg.run_idx}_it{step}.pt",
            )
        try:
            state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(
                f"Cannot load model (epoch {step}) from {model_path}"
            )
    def _load_previous_best_model(self):
        model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"model_run{self.cfg.run_idx}_best.pt",
            )
        try:
            state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(
                f"Cannot load best model from {model_path}"
            )
    def _load_previous_current_model(self):
        model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"model_run{self.cfg.run_idx}_current.pt",
            )
        try:
            state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(
                f"Cannot load current model from {model_path}"
            )
            
    def _step(self, data, step):
        # actual update step
        loss = self._batch_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.training.clip_grad_value is not None:
            # clip gradients at a certain value (this is dangerous!)
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.cfg.training.clip_grad_value,
            )
        # rescale gradients such that their norm matches a given number
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.clip_grad_norm,
                error_if_nonfinite=False,
            )
            .cpu()
            .item()
        )
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
        self.train_loss.append(loss.item())
        self.train_lr.append(self.optimizer.param_groups[0]["lr"])
        self.train_grad_norm.append(grad_norm)

        # log to mlflow
        if (
            self.cfg.use_mlflow
            and self.cfg.training.log_every_n_steps != 0
            and step % self.cfg.training.log_every_n_steps == 0
        ):
            log_dict = {
                "loss": loss.item(),
                "lr": self.train_lr[-1],
                "time_per_step": (time.time() - self.training_start_time) / (step + 1),
                "grad_norm": grad_norm,
            }
            for key, values in log_dict.items():
                log_mlflow(f"train.{key}", values, step=step)

    def _validate(self, step):
        start_time_validate = time.time()
        losses = []
        metrics = self._init_metrics()

        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            for data in self.val_loader:
                # use EMA for validation if available
                if self.ema is not None:
                    with self.ema.average_parameters():
                        loss = self._batch_loss(data)
                else:
                    loss = self._batch_loss(data)

                losses.append(loss.cpu().item())
        val_loss = np.mean(losses)
        #LOGGER.info(
        #    f"Validation loss after {step} iterations = {val_loss:.4f} "
        #    f"(mean over {len(losses)} batches)"
        #)
        if ((step + 1) % self.cfg.training.validate_every_n_steps == 0):
            self.val_loss.append(val_loss)
        if self.cfg.use_mlflow:
            log_mlflow("val.loss", val_loss, step=step)

        end_time_validate = time.time()
        #LOGGER.info(
        #    f"Validation took {end_time_validate - start_time_validate:.2f}s " ######################################################
        # )
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

    def _save_model(self, filename=None):
        if not self.cfg.save:
            return

        if filename is None:
            filename = f"model_run{self.cfg.run_idx}.pt"
        model_path = os.path.join(self.cfg.run_dir, "models", filename)
        LOGGER.debug(f"Saving model at {model_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "ema": self.ema.state_dict() if self.ema is not None else None,
            },
            model_path,
        )

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