import numpy as np
import torch

import os
import time
from omegaconf import OmegaConf, open_dict

from base_experiment import BaseExperiment
from dataset import AmplitudeDataset
from preprocessing import (
    preprocess_particles,
    preprocess_amplitude,
    undo_preprocess_amplitude,
)
from plots import plot_mixer
from logger import LOGGER
#from mlflow_util import log_mlflow
from losses import LogCoshLoss, RelL1Loss, HeteroscedasticLoss

from lloca.utils.rand_transforms import rand_lorentz
from lloca.utils.polar_decomposition import restframe_boost

import psutil, os

def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    LOGGER.info(
        f"[{tag}] RSS={mem_info.rss/1e9:.2f} GB, VMS={mem_info.vms/1e9:.2f} GB, "
        f"System available={psutil.virtual_memory().available/1e9:.2f} GB"
    )

def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        LOGGER.info(f"[{tag}] GPU allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

TYPE_TOKEN_DICT = {
    "aag": [0, 0, 1, 1, 0],
    "aag_cleaned": [0, 0, 1, 1, 0],
    "aagg": [0, 0, 1, 1, 0, 0],
    "aagg_cleaned": [0, 0, 1, 1, 0, 0],
    "zg": [0, 0, 1, 2],
    "zg_cleaned": [0, 0, 1, 2],
    "zgg": [0, 0, 1, 2, 2],
    "zgg_cleaned": [0, 0, 1, 2, 2],
    "zggg": [0, 0, 1, 2, 2, 2],
    "zggg_cleaned": [0, 0, 1, 2, 2, 2],
    "zgggg": [0, 0, 1, 2, 2, 2, 2],
    "zgggg_cleaned": [0, 0, 1, 2, 2, 2, 2],
    "zgggg_10M": [0, 0, 1, 2, 2, 2, 2],
    "zggggg": [0, 0, 1, 2, 2, 2, 2, 2],
    "wz": [0, 0, 1, 2],
    "wz_cleaned": [0, 0, 1, 2],
    "wzg": [0, 0, 1, 2, 3],
    "wzg_cleaned": [0, 0, 1, 2, 3],
    "wzgg": [0, 0, 1, 2, 3, 3],
    "wzgg_cleaned": [0, 0, 1, 2, 3, 3],
    "wwz": [0, 0, 1, 1, 2],
    "wwz_cleaned": [0, 0, 1, 1, 2],
    "qq_tth": [0, 0, 1, 1, 2],
    "qq_tth_16M": [0, 0, 1, 1, 2],
    "qq_tth_loop": [0, 1, 2, 3, 4],
    "gg_tth": [0, 0, 1, 1, 2],
    "gg_tth_loop": [0, 0, 1, 1, 2],
    "gggh": [0, 0, 1, 2],
    "qq_tth_uw": [0, 0, 1, 1, 2],
    "qq_tth_loop_uw": [0, 1, 2, 3, 4],
    "gg_tth_uw": [0, 0, 1, 1, 2],
    "gg_tth_loop_uw": [0, 0, 1, 1, 2],
    "gggh_uw": [0, 0, 1, 2],
}
DATASET_TITLE_DICT = {
    "aag": r"$gg\to\gamma\gamma g$",
    "aag_cleaned": r"$gg\to\gamma\gamma g$",
    "aag_inv": r"$gg\to\gamma\gamma g$",
    "aag_inv_naiv": r"$gg\to\gamma\gamma g$",
    "aag_inv_all": r"$gg\to\gamma\gamma g$",
    "aagg": r"$gg\to\gamma\gamma gg$",
    "aagg_cleaned": r"$gg\to\gamma\gamma gg$",
    "zg": r"$q\bar q\to Zg$",
    "zg_cleaned": r"$q\bar q\to Zg$",
    "zgg": r"$q\bar q\to Zgg$",
    "zgg_cleaned": r"$q\bar q\to Zgg$",
    "zgg_inv": r"$q\bar q\to Zgg$",
    "zgg_pinv": r"$q\bar q\to Zgg$",
    "zggg": r"$q\bar q\to Zggg$",
    "zggg_cleaned": r"$q\bar q\to Zggg$",
    "zgggg": r"$q\bar q\to Zgggg$",
    "zgggg_cleaned": r"$q\bar q\to Zgggg$",
    "zgggg_10M": r"$q\bar q\to Zgggg$",
    "zgggg_sorted": r"$q\bar q\to Zgggg$",
    "zggggg": r"$q\bar q \to Zggggg$",
    "wz": r"$q\bar q \to WZ$",
    "wz_cleaned": r"$q\bar q \to WZ$",
    "wzg": r"$q\bar q \to WZg$",
    "wzg_cleaned": r"$q\bar q \to WZg$",
    "wzgg": r"$q\bar q \to WZgg$",
    "wzgg_cleaned": r"$q\bar q \to WZgg$",
    "wwz": r"$q\bar q \to WWZ$",
    "wwz_cleaned": r"$q\bar q \to WWZ$",
    "qq_tth": r"$q\bar q \to ttH$",
    "qq_tth_16M": r"$q\bar q \to ttH$",
    "qq_tth_loop": r"$q\bar q \to ttH$ (Loop)",
    "gg_tth": r"$gg \to ttH$",
    "gg_tth_loop": r"$gg \to ttH$ (Loop)",
    "gggh": r"$gg \to Hg$",
    "qq_tth_uw": r"$q\bar q \to ttH$",
    "qq_tth_loop_uw": r"$q\bar q \to ttH$ (Loop)",
    "gg_tth_uw": r"$gg \to ttH$",
    "gg_tth_loop_uw": r"$gg \to ttH$ (Loop)",
    "gggh_uw": r"$gg \to Hg$",
}
MODEL_TITLE_DICT = {
    "Transformer": "Tr",
    "MLP": "MLP",
    "MuMLP": "μP MLP",
    "FV_MLP": "FV MLP",
    "DSI": "DSI",
    "LGATr": "LGATr",
    "LLOCATransformer": "LLoCa Transformer"
}

mass_Z = 91.188

def get_mass(dataset):
    # initialize massless particles
    mass = [1e-5] * (len(dataset))

    # Set the Z mass (change later for more general datasets)
    mass[2] = mass_Z
    return mass


class AmplitudeExperiment(BaseExperiment):
    def init_physics(self):
        assert (
            not self.cfg.training.force_xformers
        ), "amplitudes experiment assumes default torch attention"
        self.n_datasets = len(self.cfg.data.dataset)

        # create type_token list
        self.type_token = []
        for dataset in self.cfg.data.dataset:
            if self.cfg.data.include_permsym:
                self.type_token.append(TYPE_TOKEN_DICT[dataset])
            else:
                self.type_token.append(list(range(len(TYPE_TOKEN_DICT[dataset]))))

        token_size = max(
            [max([max(token) for token in self.type_token]) + 1, self.n_datasets]
        )
        OmegaConf.set_struct(self.cfg, True)
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        if self.modelname in ["GAP", "MLP", "DSI"]:
            assert len(self.cfg.data.dataset) == 1, (
                f"Architecture {self.modelname} can not handle several datasets "
                f"as specified in {self.cfg.data.dataset}"
            )

        if self.modelname == "LLOCATransformer":
            self.cfg.model.net.in_channels = token_size + 4
            self.cfg.model.net.num_scalars = token_size
        else:
            with open_dict(self.cfg):
                if self.modelname == "LGATr":
                    self.cfg.model.net.in_s_channels = token_size
                    self.cfg.model.token_size = token_size
                    
                self.cfg.model.net.type_token_list = TYPE_TOKEN_DICT[
                    self.cfg.data.dataset[0]
                ]
                assert (
                    len(np.unique(self.cfg.model.net.type_token_list))
                    == max(self.cfg.model.net.type_token_list) + 1
                ), f"Invalid type_token_list={self.cfg.model.net.type_token_list}"

    def init_data(self):
        LOGGER.info(
            f"Working with dataset {self.cfg.data.dataset} "
            f"and type_token={self.type_token}"
        )

        # load all datasets and organize them in lists
        (
            self.particles,
            self.amplitudes,
            self.particles_prepd,
            self.amplitudes_prepd,
            self.prepd_mean,
            self.prepd_std,
            self.props,
        ) = ([], [], [], [], [], [], [])
        for dataset in self.cfg.data.dataset:
            # load data
            data_path = os.path.join(self.cfg.data.data_path, f"{dataset}.npy")
            data_path_test = os.path.join(self.cfg.data.data_path, f"{dataset}_test.npy")
            data_path_val = os.path.join(self.cfg.data.data_path, f"{dataset}_val.npy")
            
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            if os.path.exists(data_path_val):
                data_train_raw = np.load(data_path)
                data_val_raw = np.load(data_path_val)
                data_test_raw = np.load(data_path_test)
                data_raw = np.concatenate([data_train_raw, data_val_raw, data_test_raw], axis=0)
            else:
                data_raw = np.load(data_path)
            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

            # bring data into correct shape
            # if self.cfg.data.subsample is not None:
            #     if self.cfg.data.subsample < self.cfg.data.train_test_val[0]*data_raw.shape[0]:
            #         LOGGER.info(
            #             f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}"
            #         )
            #         data_raw = data_raw[: int(self.cfg.data.subsample/self.cfg.data.train_test_val[0]), :]
            # else:
            #     self.cfg.data.subsample = self.cfg.data.train_test_val[0]*data_raw.shape[0]

            if os.path.exists(data_path_val):
                particles = data_raw[:, :-2]
                amplitudes = data_raw[:, [-1]]
            else:                
                particles = data_raw[:, :-1]
                amplitudes = data_raw[:, [-1]]
            assert particles.shape[1] % 4 == 0, (
                f"Invalid particle shape {particles.shape}, "
                f"last dimension must be multiple of 4"
            ) 
            # ensure that fvs are included if model is DSI or FV_MLP
            if (
                "DSI" in self.cfg.model.net._target_
                or "FV_MLP" in self.cfg.model.net._target_
            ):
                assert self.cfg.data.incl_fvs, "DSI/FV_MLP model requires fvs"

            # preprocess data
            LOGGER.info(f"Preprocessing amplitudes using trafos={self.cfg.data.amp_trafos}")
            amplitudes_prepd, prepd_mean, prepd_std = preprocess_amplitude(
                amplitudes, trafos=self.cfg.data.amp_trafos
            )
            if self.modelname == "LLOCATransformer": 
                particles = torch.tensor(particles, dtype=torch.float64)
                particles = particles.reshape(-1, len(self.type_token[0]), 4)
                mass = get_mass(self.type_token[0])
                mass = torch.tensor(mass, dtype=particles.dtype).unsqueeze(0)
                particles[..., 0] = torch.sqrt((particles[..., 1:] ** 2).sum(dim=-1) + mass**2)

                # boost to the center-of-mass ref. frame of incoming particles
                # then apply general Lorentz trafo L=R*B
                lab_particles = particles[..., :2, :].sum(dim=-2)
                to_com = restframe_boost(lab_particles)
                trafo = rand_lorentz(
                    particles.shape[:-2], generator=None, dtype=particles.dtype
                )
                trafo = torch.einsum("...ij,...jk->...ik", trafo, to_com)
                particles = torch.einsum("...ij,...kj->...ki", trafo, particles)
                particles_prepd = particles / particles.std()

                self.mom_mean = particles_prepd.mean()
                self.mom_std = np.clip(particles_prepd.std(), 1e-2, None)
                particles_prepd = particles_prepd.numpy()
            else:
                LOGGER.info(f"Preprocessing particles using trafos={self.cfg.data.trafos}")
                particles_prepd = preprocess_particles(
                    particles,
                    self.type_token[0],
                    trafos=self.cfg.data.trafos,
                    incl_fvs=self.cfg.data.incl_fvs,
                )
                #if 'LGATr' in self.cfg.model.net._target_:
                #    particles_prepd = particles_prepd.reshape(particles_prepd.shape[0], particles_prepd.shape[1] // 4, 4)

                # save number of features for later
                self.cfg.model.net.n_features = particles_prepd.shape[-1]

            # collect everything
            self.particles.append(particles)
            self.amplitudes.append(amplitudes)
            self.particles_prepd.append(particles_prepd)
            self.amplitudes_prepd.append(amplitudes_prepd)
            self.prepd_mean.append(prepd_mean)
            self.prepd_std.append(prepd_std)

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # seperate data into train, test and validation subsets for each dataset
        train_sets, test_sets, val_sets = (
            {"particles": [], "amplitudes": []},
            {"particles": [], "amplitudes": []},
            {"particles": [], "amplitudes": []},
        )
        for idataset in range(self.n_datasets):
            n_data = self.particles[idataset].shape[0]
            # desired train size
            if self.cfg.data.subsample is None:
                self.cfg.data.subsample = int(n_data*self.cfg.data.train_test_val[0])
            n_train = int(self.cfg.data.subsample) if self.cfg.data.subsample is not None else int(n_data*self.cfg.data.train_test_val[0])
            n_train = min(n_train, int(n_data*self.cfg.data.train_test_val[0]))  # cap at dataset size

            # ratio val/train relative to config
            val_ratio = self.cfg.data.train_test_val[2] / self.cfg.data.train_test_val[0]
            n_val = max(int(n_train * val_ratio),1)

            # pick indices
            train_idx = np.arange(0, n_train)
            val_idx   = np.arange(n_train, n_train + n_val)
            test_idx  = np.arange(n_train + n_val, n_data)  # remainder

            # slice preprocessed
            train_sets["particles"].append(self.particles_prepd[idataset][train_idx])
            train_sets["amplitudes"].append(self.amplitudes_prepd[idataset][train_idx])

            val_sets["particles"].append(self.particles_prepd[idataset][val_idx])
            val_sets["amplitudes"].append(self.amplitudes_prepd[idataset][val_idx])

            test_sets["particles"].append(self.particles_prepd[idataset][test_idx])
            test_sets["amplitudes"].append(self.amplitudes_prepd[idataset][test_idx])
            if self.cfg.data.no_props:
                self.props_val = self.props[idataset][self.split_test : self.split_val]

        # create dataloaders
        self.cfg.training.batchsize=int(min(self.cfg.training.batchsize,n_train/2))
        self.train_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(
                train_sets["particles"], train_sets["amplitudes"], dtype=self.dtype
            ),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(
                test_sets["particles"], test_sets["amplitudes"], dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        self.val_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(
                val_sets["particles"], val_sets["amplitudes"], dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        n_train = sum(len(x) for x in train_sets["particles"])
        n_val   = sum(len(x) for x in val_sets["particles"])
        n_test  = sum(len(x) for x in test_sets["particles"])

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_samples={n_train}, val_samples={n_val}, test_samples={n_test}, "
            f"train_batches={len(self.train_loader)}, val_batches={len(self.val_loader)}, test_batches={len(self.test_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def evaluate(self):
        with torch.no_grad():
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results_train = self._evaluate_single(
                        self.train_loader, "train"
                    )
                    self.results_val = self._evaluate_single(self.val_loader, "val")
                    self.results_test = self._evaluate_single(self.test_loader, "test")

                # also evaluate without ema to see the effect
                self._evaluate_single(self.train_loader, "train_noema")
                self._evaluate_single(self.val_loader, "val_noema")
                self._evaluate_single(self.test_loader, "test_noema")

            else:
                self.results_train = self._evaluate_single(self.train_loader, "train")
                self.results_val = self._evaluate_single(self.val_loader, "val")
                self.results_test = self._evaluate_single(self.test_loader, "test")

            self.results = {}
            for dataset in self.results_test.keys():
                self.results[dataset] = {
                    "train": self.results_train[dataset],
                    "val": self.results_val[dataset],
                    "test": self.results_test[dataset],
                }
            
        return self.results

    def call_model_fn(self, x, idataset):
        return self.model(
            x.to(self.device),
            type_token=torch.tensor(
                [self.type_token[idataset]],
                dtype=torch.long,
                device=self.device,
            ),
            global_token=torch.tensor(
                [idataset],
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _evaluate_single(self, loader, title):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        amplitudes_truth_prepd, amplitudes_pred_prepd = [
            [] for _ in range(self.n_datasets)
        ], [[] for _ in range(self.n_datasets)]
        if self.cfg.training.loss == "HETEROSC":
            amplitudes_sigmas = [[] for _ in range(self.n_datasets)]
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        t0 = time.time()
        for data in loader:
            for idataset, data_onedataset in enumerate(data):
                x, y = data_onedataset
                if self.modelname == "LLOCATransformer":
                    y_pred = self.model(
                        x.to(self.device),
                        type_token=torch.tensor(
                            [self.type_token[idataset]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        mean=self.mom_mean,
                        std=self.mom_std,
                    )
                    amplitudes_pred_prepd[idataset].append(y_pred.cpu().float().numpy())
                else:
                    if self.modelname=="LGATr":    
                        x = x.unsqueeze(0) 
                    pred = self.model(
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
                    #print(pred)
                    y_pred = pred.squeeze(0)  
                    #y_pred = pred[0, ..., 0]
                    #print(y_pred)
                    #print(f"y_pred shape: {y_pred.shape}")
                    #print(f"y shape: {y.shape}")
                    amplitudes_pred_prepd[idataset].append(y_pred[:, :1].cpu().float().numpy())
                    if self.cfg.training.loss == "HETEROSC":
                        amplitudes_sigmas[idataset].append(y_pred[:, -1:].cpu().float().numpy())
                
                amplitudes_truth_prepd[idataset].append(
                    y.cpu().float().numpy()
                )
        #print(amplitudes_pred_prepd)
        #print('amp_pred_prep shape: ', amplitudes_pred_prepd.shape)
        #if modelname=="LGATr":
        amplitudes_pred_prepd = [
            np.concatenate(individual) for individual in amplitudes_pred_prepd
        ]
        amplitudes_truth_prepd = [
            np.concatenate(individual) for individual in amplitudes_truth_prepd
        ]
        if self.cfg.training.loss == "HETEROSC":
            amplitudes_sigmas = [
                np.concatenate(individual) for individual in amplitudes_sigmas
            ]
        #print(amplitudes_pred_prepd)
        #print('amp_pred_prep shape: ', amplitudes_pred_prepd.shape)
        
        #print('amp_truth_prep shape: ', amplitudes_truth_prepd.shape)
        
        dt = (
            (time.time() - t0)
            * 1e6
            / sum(arr.shape[0] for arr in amplitudes_truth_prepd)
        )
        LOGGER.info(
            f"Evaluation time: {dt:.2f}s for 1M events "
            f"using batchsize {self.cfg.evaluation.batchsize}"
        )

        results = {}
        for idataset, dataset in enumerate(self.cfg.data.dataset):
            amp_pred_prepd = amplitudes_pred_prepd[idataset]
            amp_truth_prepd = amplitudes_truth_prepd[idataset]
            if self.cfg.training.loss == "HETEROSC":
                amplitudes_sigmas = amplitudes_sigmas[idataset]
            print(amp_pred_prepd.shape, amp_truth_prepd.shape)
            print(amp_pred_prepd.dtype, amp_truth_prepd.dtype)
            # compute metrics over preprocessed amplitudes
            #print(f"amp_pred_prepd shape: {amp_pred_prepd.shape}")
            #print(f"amp_truth_prepd shape: {amp_truth_prepd.shape}")
            mse_prepd = np.mean((amp_pred_prepd - amp_truth_prepd) ** 2)
            LOGGER.info(f"MSE on {title} {dataset} dataset: {mse_prepd:.4e}")
            l1_prepd = np.mean(np.abs(amp_pred_prepd - amp_truth_prepd))
            LOGGER.info(f"L1 on {title} {dataset} dataset: {l1_prepd:.4e}")
            l1_rel_prepd = np.mean(np.abs(amp_pred_prepd - amp_truth_prepd)/np.maximum(np.abs(amp_truth_prepd),1e-8))
            LOGGER.info(f"Relative L1 on {title} {dataset} dataset: {l1_rel_prepd:.4e}")
            
            # undo preprocessing
            amp_truth = undo_preprocess_amplitude(
                amp_truth_prepd,
                self.prepd_mean[idataset],
                self.prepd_std[idataset],
                trafos=self.cfg.data.amp_trafos,
            )
            amp_pred = undo_preprocess_amplitude(
                amp_pred_prepd,
                self.prepd_mean[idataset],
                self.prepd_std[idataset],
                trafos=self.cfg.data.amp_trafos,
            )
            if self.cfg.data.no_props:
                match title:
                    case "train":
                        amp_pred /= self.props_train.flatten()
                        amp_truth /= self.props_train.flatten()
                    case "val":
                        amp_pred /= self.props_val.flatten()
                        amp_truth /= self.props_val.flatten()
                    case "test":
                        amp_pred /= self.props_test.flatten()
                        amp_truth /= self.props_test.flatten()

            # compute metrics over actual amplitudes
            mse = np.mean((amp_truth - amp_pred) ** 2)
            l1 = np.mean(np.abs(amp_truth - amp_pred))
            l1_rel = np.mean(np.abs(amp_truth - amp_pred) / np.abs(amp_truth))

            delta = (amp_truth - amp_pred) / amp_truth
            delta_abs = np.abs(delta)
            delta_abs_mean = np.mean(delta_abs)
            LOGGER.info(
                f"Mean absolute relative error on {dataset} {title} dataset: {delta_abs_mean:.4f}"
            )
            delta_maxs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            delta_rates = []
            for delta_max in delta_maxs:
                rate = np.mean(
                    (delta > -delta_max) * (delta < delta_max)
                )  # fraction of events with -delta_max < delta < delta_max
                delta_rates.append(rate)
            LOGGER.info(
                f"rate of events in delta interval on {dataset} {title} dataset:\t"
                f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}"
            )
            # determine 1% largest amplitudes
            scale = np.abs(amp_truth)
            # log_memory_usage("before argsort")
            # log_gpu_memory("before argsort")

            #k = min(int(0.01 * len(scale)), len(scale))  # top 1% or all elements if fewer
            #idx = np.argpartition(scale, -k)[-k:]
            #LOGGER.info("Starting argpartition on %d elements", len(scale))
            idx = np.argsort(scale)
            #LOGGER.info("Finished argpartition")

            # log_memory_usage("after argpartition")
            # log_gpu_memory("after argpartition")

            delta_abs_mean_1percent = np.mean(delta_abs[idx])
            LOGGER.info(
                f"Mean absolute relative error on 1%% largest amplitudes on {dataset} {title} dataset: "
                f"{delta_abs_mean_1percent:.4f}"
            )
            if self.cfg.training.loss == "HETEROSC":
                pull = (amp_truth_prepd - amp_pred_prepd) / amplitudes_sigmas
            # log to mlflow
            if self.cfg.use_mlflow:
                log_dict = {
                    f"eval.{title}.mse": mse_prepd,
                    f"eval.{title}.l1": l1_prepd,
                    f"eval.{title}.l1_rel": l1_rel_prepd,
                    f"eval.{title}.mse_raw": mse,
                    f"eval.{title}.l1_raw": l1,
                    f"eval.{title}.l1_rel_raw": l1_rel,
                    f"eval.{title}.delta_abs_mean": delta_abs_mean,
                    f"eval.{title}.delta_abs_mean_1percent": delta_abs_mean_1percent,
                }
                for key, value in log_dict.items():
                    log_mlflow(key, value)

            amp = {
                "raw": {
                    "truth": amp_truth,
                    "prediction": amp_pred,
                    "mse": mse,
                    "l1": l1,
                    "l1_rel": l1_rel,
                },
                "preprocessed": {
                    "truth": amp_truth_prepd,
                    "prediction": amp_pred_prepd,
                    "mse": mse_prepd,
                    "l1": l1_prepd,
                    "l1_rel": l1_rel_prepd,
                },
            }
            if self.cfg.training.loss == "HETEROSC":
                amp["preprocessed"]["sigmas"] = amplitudes_sigmas
                amp["preprocessed"]["pull"] = pull
            results[dataset] = amp
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        dataset_titles = [
            DATASET_TITLE_DICT[dataset] for dataset in self.cfg.data.dataset
        ]
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = [f"{model_title}: {dataset_title}" for dataset_title in dataset_titles]
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_test"] = self.results_test
            plot_dict["results_train"] = self.results_train
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        match self.cfg.training.loss:
            case "MSE":
                self.loss = torch.nn.MSELoss()
                LOGGER.info("Using MSE loss")
            case "L1":
                self.loss = torch.nn.L1Loss()
                LOGGER.info("Using L1 loss")
            case "LogCosh":
                self.loss = LogCoshLoss()
                LOGGER.info("Using LogCosh loss")
            case "RelL1":
                self.loss = RelL1Loss()
                LOGGER.info("Using RelL1 loss")
            case "HETEROSC":
                self.loss = HeteroscedasticLoss()
                LOGGER.info("Using Heteroscedastic loss")
            case _:
                raise ValueError(f"Unknown loss function {self.cfg.training.loss}")
            
    def _init_regularization(self):
        self.regularization_lambda = self.cfg.training.regularization_lambda
        match self.cfg.training.regularization:
            case "L2":
                self.regularization = lambda model: sum(param.pow(2.0).sum() for param in model.parameters())
            case "L1":
                self.regularization = lambda model: sum(param.abs().sum() for param in model.parameters())
            case None:
                self.regularization = lambda model: 0.0
            case _:
                raise ValueError(
                    f"Unknown regularization function {self.cfg.training.regularization}"
                )

    def _batch_loss(self, data):
        # average over contributions from different datasets
        loss = 0.0
        mse = []
        if len(data) == 1:
            x, y = data[0]
            if self.modelname=="LGATr":
                x, y = x.unsqueeze(0), y.unsqueeze(0)
            attn_mask = None
            type_token = torch.tensor(
                self.type_token, dtype=torch.long, device=self.device
            )
            global_token = torch.tensor([0], dtype=torch.long, device=self.device)
        else:
            particles_max = data[-1][0].shape[-2]
            assert particles_max == max([d[0].shape[-2] for d in data])
            y = torch.stack([d[1] for d in data], dim=0)
            x = torch.zeros(len(data), *data[-1][0].shape, dtype=self.dtype)
            # carefully construct padding attention mask as float (and not bool!)
            # bool padding mask does not work, because a full row/column of zeros yields nan's in the softmax
            # this will hopefully be fixed soon, see https://github.com/pytorch/pytorch/issues/103749
            attn_mask = (
                torch.ones(
                    len(data),
                    1,
                    1,
                    1 + particles_max,
                    1 + particles_max,
                    dtype=self.dtype,
                )
                * torch.finfo(self.dtype).min
            )
            type_token = torch.zeros(
                len(data), particles_max, dtype=torch.long, device=self.device
            )
            for i, d in enumerate(data):
                particles_i = d[0].shape[-2]
                x[i, :, :particles_i, :] = d[0]
                attn_mask[i, :, :, : (1 + particles_i), : (1 + particles_i)] = 0.0
                type_token[i, :particles_i] = torch.tensor(
                    self.type_token[i], dtype=torch.long, device=self.device
                )
            global_token = torch.tensor(
                range(len(data)), dtype=torch.long, device=self.device
            )
        x, y = x.to(self.device), y.to(self.device)
        #print('_batch_loss: x shape:', x.shape, 'y shape:', y.shape)
        if self.modelname == "LLOCATransformer":
            y_pred = self.model(
                x, type_token=type_token, mean=self.mom_mean, std=self.mom_std
            )
        else:
            y_pred = self.model(
                x, type_token=type_token, global_token=global_token, attn_mask=attn_mask
            )
        if self.cfg.training.loss == "HETEROSC":
            #LOGGER.info('total output shape:', y_pred.shape)
            # Heteroscedastic loss expects the last channel to be the sigma
            #print('total output shape:', y_pred.shape)
            sigma = y_pred[..., -self.cfg.model.net.out_shape:]
            #LOGGER.info('sigma shape:', sigma.shape)
            #print('sigma shape:', sigma.shape)
            y_pred = y_pred[..., :-self.cfg.model.net.out_shape]
            #LOGGER.info('y_pred shape:', y_pred.shape)
            #print('y_pred shape:', y_pred.shape)
            loss = self.loss(y_pred, y, sigma)
        else:
            loss = self.loss(y_pred, y)
        loss = loss + self.regularization_lambda*self.regularization(self.model)
        assert torch.isfinite(loss).all()

        return loss

    def _init_metrics(self):
        metrics = {f"{dataset}.mse": [] for dataset in self.cfg.data.dataset}
        return metrics
