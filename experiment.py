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
from dataset import AmplitudeDataset, collate_variable_length

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


DATASET_TITLE_DICT = {
    "qqbar_Zg_13000GeV_amplitudes": r"$q\bar q\to Zg$",
    "ee_uu__pp_Zj": r"$e^+e^-\to u\bar u$ and $pp\to Zj$",
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
    "LLOCATransformer": "LLoCa Transformer",
    "LLOCAMuPTransformer": "LLoCa muP Transformer"
}



# def get_mass(dataset):
#     # initialize massless particles
#     mass = [1e-5] * (len(dataset))

#     # Set the Z mass (change later for more general datasets)
#     mass[2] = mass_Z
#     return mass


class AmplitudeExperiment(BaseExperiment):
    def init_physics(self):
        assert not self.cfg.training.force_xformers

        self.n_datasets = len(self.cfg.data.dataset)

        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

        if self.modelname in ["GAP", "MLP", "DSI"]:
            assert len(self.cfg.data.dataset) == 1, (
                f"Architecture {self.modelname} cannot handle several datasets"
            )

        # initialise tokenizer — will be populated in init_data
        from particle_ids import ParticleTokenizer
        tokenizer_path = os.path.join(self.cfg.run_dir, "particle_tokenizer.json")
        if self.warm_start and os.path.exists(tokenizer_path):
            self.tokenizer = ParticleTokenizer.load(tokenizer_path)
            LOGGER.info(f"Loaded tokenizer with vocab_size={self.tokenizer.vocab_size}")
        else:
            self.tokenizer = ParticleTokenizer()

        # mom_mean and mom_std are per-dataset, populated in init_data
        self.mom_mean = []
        self.mom_std  = []

        if self.modelname not in ("LLOCATransformer", "LLOCAMuPTransformer"):
            self.type_token = []
            for dataset in self.cfg.data.dataset:
                if self.cfg.data.include_permsym:
                    self.type_token.append(TYPE_TOKEN_DICT[dataset])
                else:
                    self.type_token.append(list(range(len(TYPE_TOKEN_DICT[dataset]))))
        
    def init_data(self):
        LOGGER.info(f"Loading datasets: {self.cfg.data.dataset}")
    
        (
            self.particles,
            self.amplitudes,
            self.particles_prepd,
            self.amplitudes_prepd,
            self.prepd_mean,
            self.prepd_std,
            self.pdg_ids,
            self.mom_mean,
            self.mom_std,
        ) = ([], [], [], [], [], [], [], [], [])
    
        for dataset in self.cfg.data.dataset:
            data_path = os.path.join(self.cfg.data.data_path, f"{dataset}.npy")
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            data_raw = np.load(data_path)
    
            # shuffle with fixed seed to break any ordering from the pipeline
            rng = np.random.default_rng(seed=42)
            rng.shuffle(data_raw, axis=0)
    
            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")
    
            # layout: (N, nparticles*4 + nparticles + 1)
            n_particles  = (data_raw.shape[1] - 1) // 5
            momenta_cols = n_particles * 4
    
            particles  = data_raw[:, :momenta_cols]
            pdg_ids    = data_raw[:, momenta_cols:-1].astype(int)
            amplitudes = data_raw[:, [-1]]
    
            # encode PDG ids to contiguous token indices
            type_tokens = self.tokenizer.register_and_encode(pdg_ids)  # (N, n_particles)
    
            # preprocess amplitude
            LOGGER.info(f"Preprocessing amplitudes using trafos={self.cfg.data.amp_trafos}")
            amplitudes_prepd, prepd_mean, prepd_std = preprocess_amplitude(
                amplitudes, trafos=self.cfg.data.amp_trafos
            )
    
            # preprocess particles
            if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer"):
                particles_t = torch.tensor(particles, dtype=torch.float64)
                particles_t = particles_t.reshape(-1, n_particles, 4)
    
                # recompute E from 3-momentum and on-shell mass
                m2 = particles_t[..., 0] ** 2 - (particles_t[..., 1:] ** 2).sum(dim=-1)
                particles_t[..., 0] = torch.sqrt(
                    (particles_t[..., 1:] ** 2).sum(dim=-1) + m2.clamp(min=0)
                )
    
                lab_particles = particles_t[..., :2, :].sum(dim=-2)
                to_com = restframe_boost(lab_particles)
                trafo = rand_lorentz(
                    particles_t.shape[:-2], generator=None, dtype=particles_t.dtype
                )
                trafo = torch.einsum("...ij,...jk->...ik", trafo, to_com)
                particles_t = torch.einsum("...ij,...kj->...ki", trafo, particles_t)
    
                particles_prepd = particles_t / particles_t.std()
                self.mom_mean.append(float(particles_prepd.mean()))
                self.mom_std.append(float(np.clip(particles_prepd.std(), 1e-2, None)))
                particles_prepd = particles_prepd.numpy()  # (N, n_particles, 4)
            else:
                LOGGER.info(f"Preprocessing particles using trafos={self.cfg.data.trafos}")
                particles_prepd = preprocess_particles(
                    particles,
                    type_tokens[0],
                    trafos=self.cfg.data.trafos,
                    incl_fvs=self.cfg.data.incl_fvs,
                )
    
            self.particles.append(particles)
            self.amplitudes.append(amplitudes)
            self.particles_prepd.append(particles_prepd)
            self.amplitudes_prepd.append(amplitudes_prepd)
            self.prepd_mean.append(prepd_mean)
            self.prepd_std.append(prepd_std)
            self.pdg_ids.append(type_tokens)
    
        # ------------------------------------------------------------------
        # Pad shorter datasets and concatenate into one flat array
        # ------------------------------------------------------------------
        # nparticles per dataset — inferred from the flat momenta shape
        is_lloca = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer")
        nparticles_list = [
            p.shape[1] if is_lloca else p.shape[1] // 4
            for p in self.particles_prepd
        ]
        nparticles_max = max(nparticles_list)
        LOGGER.info(
            f"nparticles per dataset: {nparticles_list} → padding to {nparticles_max}"
        )
    
        padded_particles  = []
        padded_tokens     = []
        padded_amplitudes = []
    
        for i, (parts, toks, amps) in enumerate(
            zip(self.particles_prepd, self.pdg_ids, self.amplitudes_prepd)
        ):
            N     = parts.shape[0]
            n_i   = nparticles_list[i]
            n_pad = nparticles_max - n_i
    
            if n_pad == 0:
                padded_particles.append(parts)
                padded_tokens.append(toks)
            else:
                if is_lloca:
                    # parts shape: (N, n_i, 4)
                    pad_p = np.zeros((N, n_pad, 4), dtype=parts.dtype)
                    padded_particles.append(np.concatenate([parts, pad_p], axis=1))
                else:
                    # parts shape: (N, n_i * 4)
                    pad_p = np.zeros((N, n_pad * 4), dtype=parts.dtype)
                    padded_particles.append(np.concatenate([parts, pad_p], axis=1))
                # token 0 is the reserved padding index
                pad_t = np.zeros((N, n_pad), dtype=toks.dtype)
                padded_tokens.append(np.concatenate([toks, pad_t], axis=1))
    
            padded_amplitudes.append(amps)
    
        self.all_particles  = np.concatenate(padded_particles,  axis=0)
        self.all_tokens     = np.concatenate(padded_tokens,     axis=0)
        self.all_amplitudes = np.concatenate(padded_amplitudes, axis=0)
        self.nparticles_max = nparticles_max
        LOGGER.info(f"Combined dataset: {self.all_particles.shape[0]} events")
    
        # keep a single prepd_mean/std for the combined amplitude (all datasets
        # were preprocessed jointly — we just use index 0 since the trafos are
        # stateless; for standardization this would need revisiting)
        # For now store them all; _evaluate_single uses index 0 for the combined set.
    
        # ------------------------------------------------------------------
        # Set model size now that the tokenizer vocab is finalised
        # ------------------------------------------------------------------
        token_size = self.tokenizer.vocab_size
        self.token_size = token_size
        with open_dict(self.cfg):
            self.cfg.model.token_size = token_size
        if is_lloca:
            with open_dict(self.cfg):
                self.cfg.model.net.in_channels = token_size + 4
                self.cfg.model.net.num_scalars = token_size
        else:
            with open_dict(self.cfg):
                if self.modelname == "LGATr":
                    self.cfg.model.net.in_s_channels = token_size
    
        self.cfg.model.net.n_features = self.all_particles.shape[-1]
    
        if self.cfg.save:
            tokenizer_path = os.path.join(self.cfg.run_dir, "particle_tokenizer.json")
            self.tokenizer.save(tokenizer_path)
            LOGGER.info(f"Saved tokenizer with vocab_size={token_size}")

            
    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1
    
        n_data = self.all_particles.shape[0]
    
        if self.cfg.data.subsample is None:
            self.cfg.data.subsample = int(n_data * self.cfg.data.train_test_val[0])
    
        n_train = int(self.cfg.data.subsample)
        n_train = min(n_train, int(n_data * self.cfg.data.train_test_val[0]))
        if n_train % 2 != 0 and n_train > 1:
            n_train -= 1
        elif n_train == 1:
            n_train = 2
    
        val_ratio = self.cfg.data.train_test_val[2] / self.cfg.data.train_test_val[0]
        n_val = max(int(n_train * val_ratio), 2)
        if n_val % 2 != 0:
            n_val -= 1
    
        train_idx = np.arange(0, n_train)
        val_idx   = np.arange(n_train, n_train + n_val)
        test_idx  = np.arange(n_train + n_val, n_data)
    
        is_lloca = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer")
    
        def make_loader(indices, shuffle, batchsize):
            ds = AmplitudeDataset(
                particles  = self.all_particles[indices],
                amplitudes = self.all_amplitudes[indices],
                tokens     = self.all_tokens[indices],
                dtype      = self.dtype,
                reshape    = is_lloca,
            )
            return torch.utils.data.DataLoader(
                ds,
                batch_size=batchsize,
                shuffle=shuffle,
                drop_last=True,
                collate_fn=collate_variable_length,
            )
    
        self.cfg.training.batchsize = int(min(self.cfg.training.batchsize, n_train / 2))
    
        self.train_loader = make_loader(train_idx, shuffle=True,
                                        batchsize=self.cfg.training.batchsize)
        self.val_loader   = make_loader(val_idx,   shuffle=False,
                                        batchsize=min(self.cfg.evaluation.batchsize,
                                                    max(len(val_idx) // 2, 1)))
        self.test_loader  = make_loader(test_idx,  shuffle=False,
                                        batchsize=min(self.cfg.evaluation.batchsize,
                                                    max(len(test_idx) // 2, 1)))
    
        LOGGER.info(
            f"Constructed dataloaders: "
            f"train={n_train}, val={n_val}, test={len(test_idx)} | "
            f"batches={len(self.train_loader)}/{len(self.val_loader)}/{len(self.test_loader)} | "
            f"batchsize={self.cfg.training.batchsize} (train), "
            f"{self.cfg.evaluation.batchsize} (eval)"
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

    def call_model_fn(self, x, tokens=None):
        if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer"):
            if tokens is None:
                raise ValueError("call_model_fn requires tokens for LLoCA models")
            return self.model(
                x.to(self.device),
                type_token=tokens.to(self.device),
                mean=self.mom_mean[0],
                std=self.mom_std[0],
            )
        else:
            return self.model(
                x.to(self.device),
                type_token=torch.tensor(
                    [self.type_token[0]],
                    dtype=torch.long,
                    device=self.device,
                ),
                global_token=torch.tensor(
                    [0],
                    dtype=torch.long,
                    device=self.device,
                ),
            )

    def _evaluate_single(self, loader, title):
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
    
        is_lloca = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer")
        all_pred  = []
        all_truth = []
        all_sigma = [] if self.cfg.training.loss == "HETEROSC" else None
    
        t0 = time.time()
        for data in loader:
            x, y, tokens = data
    
            if is_lloca:
                x      = x.to(self.device)
                tokens = tokens.to(self.device)
                y_pred = self.model(
                    x,
                    type_token=tokens,
                    mean=self.mom_mean[0],
                    std=self.mom_std[0],
                )
            else:
                x      = x.to(self.device)
                tokens = tokens.to(self.device)
                B, nparticles_max = tokens.shape
                is_real = tokens != 0
                neg_inf = torch.finfo(self.dtype).min
                attn_mask = torch.full(
                    (B, 1, 1, 1 + nparticles_max, 1 + nparticles_max),
                    neg_inf, dtype=self.dtype, device=self.device,
                )
                real_with_global = torch.cat(
                    [torch.ones(B, 1, dtype=torch.bool, device=self.device), is_real], dim=1
                )
                allowed = real_with_global.unsqueeze(2) & real_with_global.unsqueeze(1)
                attn_mask[:, 0, 0, :, :] = torch.where(
                    allowed,
                    torch.zeros(1, dtype=self.dtype, device=self.device),
                    attn_mask[:, 0, 0, :, :],
                )
    
                if self.modelname == "LGATr":
                    x = x.unsqueeze(0)
    
                y_pred = self.model(
                    x,
                    type_token=tokens,
                    global_token=torch.zeros(B, dtype=torch.long, device=self.device),
                    attn_mask=attn_mask,
                )
                if self.modelname == "LGATr" and y_pred.shape[0] == 1:
                    y_pred = y_pred.squeeze(0)
    
            out_shape = (
                self.cfg.model.net.get("out_shape")
                or self.cfg.model.net.get("out_channels")
            )
            if self.cfg.training.loss == "HETEROSC":
                all_sigma.append(y_pred[..., -out_shape:].cpu().float().numpy())
                y_pred = y_pred[..., :-out_shape]
    
            all_pred.append(y_pred[:, :1].cpu().float().numpy() if not is_lloca
                            else y_pred.cpu().float().numpy())
            all_truth.append(y.cpu().float().numpy())
    
        amp_pred_prepd  = np.concatenate(all_pred,  axis=0)
        amp_truth_prepd = np.concatenate(all_truth, axis=0)
    
        dt = time.time() - t0
        LOGGER.info(
            f"Evaluation time: {dt:.2f}s for {len(amp_truth_prepd)} samples "
            f"using batchsize {self.cfg.evaluation.batchsize}"
        )
    
        # use "combined" as key when multiple datasets were merged
        result_key = (
            "combined" if len(self.cfg.data.dataset) > 1 else self.cfg.data.dataset[0]
        )
    
        mse_prepd    = np.mean((amp_pred_prepd - amp_truth_prepd) ** 2)
        l1_prepd     = np.mean(np.abs(amp_pred_prepd - amp_truth_prepd))
        l1_rel_prepd = np.mean(
            np.abs(amp_pred_prepd - amp_truth_prepd)
            / np.maximum(np.abs(amp_truth_prepd), 1e-8)
        )
        LOGGER.info(f"MSE (prepd) {title} {result_key}: {mse_prepd:.4e}")
        LOGGER.info(f"L1  (prepd) {title} {result_key}: {l1_prepd:.4e}")
        LOGGER.info(f"L1r (prepd) {title} {result_key}: {l1_rel_prepd:.4e}")
    
        amp_truth = undo_preprocess_amplitude(
            amp_truth_prepd, self.prepd_mean[0], self.prepd_std[0],
            trafos=self.cfg.data.amp_trafos,
        )
        amp_pred = undo_preprocess_amplitude(
            amp_pred_prepd, self.prepd_mean[0], self.prepd_std[0],
            trafos=self.cfg.data.amp_trafos,
        )
    
        mse    = np.mean((amp_truth - amp_pred) ** 2)
        l1     = np.mean(np.abs(amp_truth - amp_pred))
        l1_rel = np.mean(np.abs(amp_truth - amp_pred) / np.abs(amp_truth))
        delta  = (amp_truth - amp_pred) / amp_truth
        delta_abs = np.abs(delta)
        delta_abs_mean = np.mean(delta_abs)
        LOGGER.info(f"Mean |rel err| {title} {result_key}: {delta_abs_mean:.4f}")
    
        delta_maxs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        delta_rates = [
            np.mean((delta > -dm) * (delta < dm)) for dm in delta_maxs
        ]
        LOGGER.info(
            f"rate of events in delta interval on {result_key} {title}:\t"
            f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}"
        )
    
        scale = np.abs(amp_truth)
        idx   = np.argsort(scale)
        delta_abs_mean_1percent = np.mean(delta_abs[idx])
        LOGGER.info(
            f"Mean |rel err| on 1%% largest amplitudes {result_key} {title}: "
            f"{delta_abs_mean_1percent:.4f}"
        )
    
        amp = {
            "raw": {
                "truth":      amp_truth,
                "prediction": amp_pred,
                "mse":        mse,
                "l1":         l1,
                "l1_rel":     l1_rel,
            },
            "preprocessed": {
                "truth":      amp_truth_prepd,
                "prediction": amp_pred_prepd,
                "mse":        mse_prepd,
                "l1":         l1_prepd,
                "l1_rel":     l1_rel_prepd,
            },
        }
        if self.cfg.training.loss == "HETEROSC":
            sigmas = np.concatenate(all_sigma, axis=0)
            pull   = (amp_truth_prepd - amp_pred_prepd) / sigmas
            amp["preprocessed"]["sigmas"] = sigmas
            amp["preprocessed"]["pull"]   = pull
    
        return {result_key: amp}

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        dataset_titles = [
            DATASET_TITLE_DICT[dataset] for dataset in self.cfg.data.dataset
        ]
        model_core = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model_title = MODEL_TITLE_DICT[type(model_core.net).__name__]   
        #model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = [f"{model_title}: {dataset_title}" for dataset_title in dataset_titles]
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_test"] = self.results_test
            plot_dict["results_train"] = self.results_train
        if self.cfg.train:
            plot_dict["train_loss"]        = self.train_loss
            plot_dict["val_loss"]          = self.val_loss
            plot_dict["train_lr"]          = self.train_lr
            # new:
            plot_dict["train_loss_no_reg"] = getattr(self, "train_loss_no_reg", [])
            plot_dict["val_loss_no_reg"]   = getattr(self, "val_loss_no_reg",   [])
            plot_dict["train_mse"]         = getattr(self, "train_mse",         [])
            plot_dict["val_mse"]           = getattr(self, "val_mse",           [])

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
        x, y, tokens = data   # (B, nparticles_max[,4]), (B,1), (B, nparticles_max)
    
        if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer"):
            return self._batch_loss_lloca(x, y, tokens)
    
        x      = x.to(self.device)
        y      = y.to(self.device)
        tokens = tokens.to(self.device)   # (B, nparticles_max)
    
        B, nparticles_max = tokens.shape
    
        # padding mask: True where token == 0 means padding
        # real particles have token >= 1 (0 is reserved for padding by the tokenizer)
        is_real = tokens != 0   # (B, nparticles_max)
    
        # build float attention mask  (B, 1, 1, 1+P, 1+P)
        # 0.0 = attend, -inf = ignore
        neg_inf = torch.finfo(self.dtype).min
        attn_mask = torch.full(
            (B, 1, 1, 1 + nparticles_max, 1 + nparticles_max),
            neg_inf, dtype=self.dtype, device=self.device,
        )
        # global token (position 0) can always attend to itself and all real particles
        real_with_global = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=self.device), is_real], dim=1
        )  # (B, 1+P)
        allowed = real_with_global.unsqueeze(2) & real_with_global.unsqueeze(1)  # (B, 1+P, 1+P)
        attn_mask[:, 0, 0, :, :] = torch.where(allowed, torch.zeros(1, dtype=self.dtype, device=self.device), attn_mask[:, 0, 0, :, :])
    
        if self.modelname == "LGATr":
            x = x.unsqueeze(0)
    
        y_pred = self.model(
            x,
            type_token=tokens,
            global_token=torch.zeros(B, dtype=torch.long, device=self.device),
            attn_mask=attn_mask,
        )
    
        out_shape = self.cfg.model.net.get("out_shape") or self.cfg.model.net.get("out_channels")
        if self.cfg.training.loss == "HETEROSC":
            sigma   = y_pred[..., -out_shape:]
            y_pred  = y_pred[..., :-out_shape]
            loss    = self.loss(y_pred, y, sigma)
            mse_val = torch.nn.functional.mse_loss(y_pred, y).item()
        else:
            loss    = self.loss(y_pred, y)
            mse_val = None
    
        reg_term    = self.regularization_lambda * self.regularization(self.model)
        loss_no_reg = loss.item()
        loss        = loss + reg_term
        assert torch.isfinite(loss).all()
        return loss, loss_no_reg, mse_val

    def _batch_loss_lloca(self, x, y, tokens):
        # x:      (B, nparticles_max, 4)
        # y:      (B, 1)
        # tokens: (B, nparticles_max)
        # mom_mean/std are now lists (one per original dataset) but for the combined
        # dataset we use a single global normalisation stored at index 0.
        x      = x.to(self.device)
        y      = y.to(self.device)
        tokens = tokens.to(self.device)
    
        y_pred = self.model(
            x,
            type_token=tokens,
            mean=self.mom_mean[0],
            std=self.mom_std[0],
        )
    
        loss        = self.loss(y_pred, y)
        reg         = self.regularization_lambda * self.regularization(self.model)
        loss_no_reg = loss.item()
        loss        = loss + reg
        assert torch.isfinite(loss).all()
        return loss, loss_no_reg, None

    def _init_metrics(self):
        result_key = (
            "combined" if len(self.cfg.data.dataset) > 1 else self.cfg.data.dataset[0]
        )
        return {f"{result_key}.mse": []}
    
