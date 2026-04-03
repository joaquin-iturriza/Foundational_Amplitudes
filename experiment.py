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
            self.pdg_ids,
            self.mom_mean,
            self.mom_std,
        ) = ([], [], [], [], [], [])

        # Normalise per-dataset subsample spec to a list[int|None]
        subsample_cfg = self.cfg.data.subsample
        if subsample_cfg is None:
            subsample_per_ds = [None] * len(self.cfg.data.dataset)
        elif OmegaConf.is_list(subsample_cfg) or isinstance(subsample_cfg, (list, tuple)):
            subsample_per_ds = [int(s) if s is not None else None for s in subsample_cfg]
        else:
            subsample_per_ds = [int(subsample_cfg)] * len(self.cfg.data.dataset)

        for dataset, sub_n in zip(self.cfg.data.dataset, subsample_per_ds):
            data_path = os.path.join(self.cfg.data.data_path, f"{dataset}.npy")
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            data_raw = np.load(data_path)
            if sub_n is not None:
                data_raw = data_raw[:sub_n]
            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")
    
            n_particles  = (data_raw.shape[1] - 1) // 5
            momenta_cols = n_particles * 4
    
            particles  = data_raw[:, :momenta_cols]
            pdg_ids    = data_raw[:, momenta_cols:-1].astype(int)
            amplitudes = data_raw[:, [-1]]
    
            type_tokens = self.tokenizer.register_and_encode(pdg_ids)
    
            if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer"):
                particles_t = torch.tensor(particles, dtype=torch.float64)
                particles_t = particles_t.reshape(-1, n_particles, 4)
    
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
                self.mom_std.append(float(particles_prepd.std().clamp(min=1e-2)))
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
            self.pdg_ids.append(type_tokens)
    
        # ------------------------------------------------------------------
        # Build flat per-event lists and concatenate amplitudes
        # ------------------------------------------------------------------
        is_lloca = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer")
    
        all_particles_list = []
        all_tokens_list    = []
        all_amplitudes_raw = []
        all_process_ids    = []   # integer process index per event

        for proc_idx, (parts, toks, amps) in enumerate(zip(self.particles_prepd, self.pdg_ids, self.amplitudes)):
            N = parts.shape[0]
            if is_lloca:
                # parts: (N, n_particles, 4) → list of (n_particles, 4)
                all_particles_list.extend([parts[j] for j in range(N)])
            else:
                all_particles_list.extend([parts[j] for j in range(N)])
            all_tokens_list.extend([toks[j] for j in range(N)])
            all_amplitudes_raw.append(amps)
            all_process_ids.extend([proc_idx] * N)

        all_amplitudes_raw = np.concatenate(all_amplitudes_raw, axis=0)  # (N_total, 1)
        all_process_ids    = np.array(all_process_ids, dtype=np.int32)   # (N_total,)
    
        # ------------------------------------------------------------------
        # Global amplitude preprocessing across all datasets combined
        # ------------------------------------------------------------------
        LOGGER.info(
            f"Preprocessing amplitudes globally using trafos={self.cfg.data.amp_trafos}"
        )
        all_amplitudes_prepd, prepd_mean, prepd_std = preprocess_amplitude(
            all_amplitudes_raw, trafos=self.cfg.data.amp_trafos
        )
        self.prepd_mean = [prepd_mean]
        self.prepd_std  = [prepd_std]
        LOGGER.info(
            f"Combined amplitude stats after preprocessing: "
            f"mean={all_amplitudes_prepd.mean():.3f}, "
            f"std={all_amplitudes_prepd.std():.3f}, "
            f"min={all_amplitudes_prepd.min():.3f}, "
            f"max={all_amplitudes_prepd.max():.3f}"
        )
    
        # ------------------------------------------------------------------
        # Shuffle with fixed seed
        # ------------------------------------------------------------------
        N_total = len(all_particles_list)
        rng  = np.random.default_rng(seed=42)
        perm = rng.permutation(N_total)
    
        all_particles_list   = [all_particles_list[i]   for i in perm]
        all_tokens_list      = [all_tokens_list[i]      for i in perm]
        all_amplitudes_prepd = all_amplitudes_prepd[perm]
        all_process_ids      = all_process_ids[perm]
    
        # ------------------------------------------------------------------
        # Build contiguous flat arrays for fast O(1) indexing in DataLoader
        # ------------------------------------------------------------------
        from dataset import build_flat_arrays
        particles_flat, tokens_flat, offsets = build_flat_arrays(
            all_particles_list, all_tokens_list
        )
        self.particles_flat    = particles_flat       # (N_total_particles, 4)
        self.tokens_flat       = tokens_flat          # (N_total_particles,)
        self.offsets           = offsets              # (N_events, 2)
        self.all_amplitudes    = all_amplitudes_prepd # (N_events, 1)
        self.all_process_ids   = all_process_ids      # (N_events,)
        self.N_events          = N_total
    
        LOGGER.info(
            f"Combined dataset: {N_total} events, "
            f"{particles_flat.shape[0]} total particles"
        )
    
        # ------------------------------------------------------------------
        # Set model size
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
    
        if self.cfg.save:
            tokenizer_path = os.path.join(self.cfg.run_dir, "particle_tokenizer.json")
            self.tokenizer.save(tokenizer_path)
            LOGGER.info(f"Saved tokenizer with vocab_size={token_size}")
            
    def _init_dataloader(self):
        from dataset import AmplitudeDataset, collate_variable_length, ProcessBalancedSampler
        assert sum(self.cfg.data.train_test_val) <= 1

        N_total = self.N_events

        n_train = int(N_total * self.cfg.data.train_test_val[0])
        if n_train % 2 != 0 and n_train > 1:
            n_train -= 1
        elif n_train == 1:
            n_train = 2

        # Store as scalar for base_experiment._evals compatibility
        with open_dict(self.cfg):
            self.cfg.data.subsample = n_train

        val_ratio = self.cfg.data.train_test_val[2] / self.cfg.data.train_test_val[0]
        n_val = max(int(n_train * val_ratio), 2)
        if n_val % 2 != 0:
            n_val -= 1

        train_idx = np.arange(0,       n_train)
        val_idx   = np.arange(n_train, n_train + n_val)
        test_idx  = np.arange(n_train + n_val, N_total)

        nw = self.cfg.training.num_workers

        def make_dataset(indices):
            return AmplitudeDataset(
                particles_flat = self.particles_flat,
                offsets        = self.offsets[indices],
                amplitudes     = self.all_amplitudes[indices],
                tokens_flat    = self.tokens_flat,
                dtype          = self.dtype,
            )

        def make_loader(indices, shuffle, batchsize, sampler=None):
            ds = make_dataset(indices)
            return torch.utils.data.DataLoader(
                ds,
                batch_size  = batchsize,
                shuffle     = shuffle if sampler is None else False,
                sampler     = sampler,
                drop_last   = True,
                collate_fn  = collate_variable_length,
                pin_memory  = torch.cuda.is_available() and nw > 0,
                num_workers = nw,
            )

        self.cfg.training.batchsize = int(min(self.cfg.training.batchsize, n_train / 2))
        batchsize = self.cfg.training.batchsize

        # --- training loader: balanced sampler when multiple processes ---
        if self.n_datasets > 1:
            train_proc_ids = self.all_process_ids[train_idx]
            self.train_sampler = ProcessBalancedSampler(
                process_ids = train_proc_ids,
                batch_size  = batchsize,
                seed        = 0,
            )
            self.train_loader = make_loader(train_idx, shuffle=False,
                                            batchsize=batchsize,
                                            sampler=self.train_sampler)
            LOGGER.info(
                f"Using ProcessBalancedSampler across {self.n_datasets} processes "
                f"(batch_size={batchsize})"
            )
        else:
            self.train_sampler = None
            self.train_loader  = make_loader(train_idx, shuffle=True, batchsize=batchsize)

        # Plain train loader for evaluation — always reflects the true data distribution,
        # regardless of whether a balanced sampler is used for training.
        train_eval_bs = min(self.cfg.evaluation.batchsize, max(len(train_idx) // 2, 1))
        self.train_eval_loader = make_loader(train_idx, shuffle=False, batchsize=train_eval_bs)

        eval_bs = min(self.cfg.evaluation.batchsize, max(len(val_idx) // 2, 1))
        self.val_loader  = make_loader(val_idx,  shuffle=False, batchsize=eval_bs)
        self.test_loader = make_loader(test_idx, shuffle=False,
                                       batchsize=min(self.cfg.evaluation.batchsize,
                                                     max(len(test_idx) // 2, 1)))

        # --- per-process loaders (val for loss tracking; test+train for per-dataset plots) ---
        self.proc_val_loaders        = {}
        self.proc_test_loaders       = {}
        self.proc_train_eval_loaders = {}
        if self.n_datasets > 1:
            val_proc_ids   = self.all_process_ids[val_idx]
            test_proc_ids  = self.all_process_ids[test_idx]
            train_proc_ids = self.all_process_ids[train_idx]
            test_bs        = min(self.cfg.evaluation.batchsize, max(len(test_idx) // 2, 1))
            for p, name in enumerate(self.cfg.data.dataset):
                # val
                mask  = val_proc_ids == p
                p_idx = val_idx[mask]
                if len(p_idx) >= 2:
                    self.proc_val_loaders[name] = make_loader(
                        p_idx, shuffle=False,
                        batchsize=min(eval_bs, max(len(p_idx) // 2, 1))
                    )
                    LOGGER.info(f"  Per-process val loader '{name}': {len(p_idx)} events")
                # test
                mask  = test_proc_ids == p
                p_idx = test_idx[mask]
                if len(p_idx) >= 2:
                    self.proc_test_loaders[name] = make_loader(
                        p_idx, shuffle=False,
                        batchsize=min(test_bs, max(len(p_idx) // 2, 1))
                    )
                # train (plain, for evaluation only)
                mask  = train_proc_ids == p
                p_idx = train_idx[mask]
                if len(p_idx) >= 2:
                    self.proc_train_eval_loaders[name] = make_loader(
                        p_idx, shuffle=False,
                        batchsize=min(train_eval_bs, max(len(p_idx) // 2, 1))
                    )
    
        LOGGER.info(
            f"Constructed dataloaders: train={n_train}, val={n_val}, "
            f"test={len(test_idx)} | "
            f"batches={len(self.train_loader)}/{len(self.val_loader)}/"
            f"{len(self.test_loader)} | "
            f"batchsize={self.cfg.training.batchsize} (train), "
            f"{self.cfg.evaluation.batchsize} (eval)"
        )
                
    def evaluate(self):
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()

        combined_key = "combined" if len(self.cfg.data.dataset) > 1 else self.cfg.data.dataset[0]

        def collect(loader):
            with torch.no_grad():
                if self.ema is not None:
                    with self.ema.average_parameters():
                        return self._collect_predictions(loader)
                return self._collect_predictions(loader)

        # ------------------------------------------------------------------
        # Collect per-process predictions (single forward pass per process)
        # ------------------------------------------------------------------
        proc_preds = {}   # name -> {"train": (pred,truth,sig), "test": ..., "val": ...}

        loaders_by_split = [
            ("test",  self.proc_test_loaders),
            ("train", self.proc_train_eval_loaders),
            ("val",   self.proc_val_loaders),
        ]
        for split, loader_dict in loaders_by_split:
            for name, loader in loader_dict.items():
                LOGGER.info(f"### Evaluating {split} [{name}] ###")
                if name not in proc_preds:
                    proc_preds[name] = {}
                proc_preds[name][split] = collect(loader)

        # ------------------------------------------------------------------
        # Build combined arrays by concatenation (no second forward pass)
        # ------------------------------------------------------------------
        def concat_split(split):
            available = [n for n in proc_preds if split in proc_preds[n]]
            if not available:
                # single-dataset fallback: collect directly
                loader = {"train": self.train_eval_loader,
                          "val":   self.val_loader,
                          "test":  self.test_loader}[split]
                return collect(loader)
            pred   = np.concatenate([proc_preds[n][split][0] for n in available], axis=0)
            truth  = np.concatenate([proc_preds[n][split][1] for n in available], axis=0)
            first_sig = proc_preds[available[0]][split][2]
            sigmas = (np.concatenate([proc_preds[n][split][2] for n in available], axis=0)
                      if first_sig is not None else None)
            return pred, truth, sigmas

        # ------------------------------------------------------------------
        # Compute metrics (pure numpy, fast)
        # ------------------------------------------------------------------
        LOGGER.info("### Computing combined metrics ###")
        for split, attr in [("train", "results_train"), ("val", "results_val"), ("test", "results_test")]:
            pred, truth, sigmas = concat_split(split)
            setattr(self, attr,
                    self._metrics_from_arrays(pred, truth, split, combined_key, sigmas))

        self.results = {
            combined_key: {
                "train": self.results_train[combined_key],
                "val":   self.results_val[combined_key],
                "test":  self.results_test[combined_key],
            }
        }

        # Per-process results for per-dataset histogram plots
        self.results_per_proc = {}
        for name, splits in proc_preds.items():
            self.results_per_proc[name] = {}
            for split, (pred, truth, sigmas) in splits.items():
                self.results_per_proc[name][split] = self._metrics_from_arrays(
                    pred, truth, f"{split}_{name}", name, sigmas
                )[name]

        # Optionally log noema metrics (no extra forward pass — just re-use arrays)
        if self.ema is not None:
            LOGGER.info("### Evaluating without EMA (reusing predictions not possible — skipping noema) ###")

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

    def _collect_predictions(self, loader):
        """Run model forward pass over a loader.

        Caller is responsible for torch.no_grad() and EMA context.

        Returns
        -------
        amp_pred_prepd  : np.ndarray  (N, 1)
        amp_truth_prepd : np.ndarray  (N, 1)
        sigmas          : np.ndarray | None   (only for HETEROSC loss)
        """
        is_lloca  = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer")
        all_pred  = []
        all_truth = []
        all_sigma = [] if self.cfg.training.loss == "HETEROSC" else None

        t0 = time.time()
        for data in loader:
            particles, y, tokens, ptr = data
            if is_lloca:
                particles = particles.to(self.device)
                tokens    = tokens.to(self.device)
                ptr       = ptr.to(self.device)
                y_pred = self.model(
                    particles, tokens,
                    mean=self.mom_mean[0], std=self.mom_std[0],
                    ptr=ptr,
                )
            else:
                x      = particles.to(self.device)
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

        LOGGER.info(
            f"Collected {sum(len(p) for p in all_pred)} predictions in {time.time()-t0:.2f}s"
        )
        amp_pred_prepd  = np.concatenate(all_pred,  axis=0)
        amp_truth_prepd = np.concatenate(all_truth, axis=0)
        sigmas = np.concatenate(all_sigma, axis=0) if all_sigma else None
        return amp_pred_prepd, amp_truth_prepd, sigmas

    def _metrics_from_arrays(self, amp_pred_prepd, amp_truth_prepd, title, result_key,
                             sigmas=None):
        """Compute metrics from preprocessed arrays (no model call). Pure numpy."""
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
        delta     = (amp_truth - amp_pred) / amp_truth
        delta_abs = np.abs(delta)
        LOGGER.info(f"Mean |rel err| {title} {result_key}: {np.mean(delta_abs):.4f}")

        delta_maxs  = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        delta_rates = [np.mean((delta > -dm) * (delta < dm)) for dm in delta_maxs]
        LOGGER.info(
            f"rate of events in delta interval on {result_key} {title}:\t"
            f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}"
        )

        idx = np.argsort(np.abs(amp_truth))
        LOGGER.info(
            f"Mean |rel err| on 1%% largest amplitudes {result_key} {title}: "
            f"{np.mean(delta_abs[idx]):.4f}"
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
        if sigmas is not None:
            pull = (amp_truth_prepd - amp_pred_prepd) / sigmas
            amp["preprocessed"]["sigmas"] = sigmas
            amp["preprocessed"]["pull"]   = pull

        return {result_key: amp}

    def _evaluate_single(self, loader, title):
        """Thin wrapper kept for backward compatibility (e.g. base_experiment._evals)."""
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        result_key = (
            "combined" if len(self.cfg.data.dataset) > 1 else self.cfg.data.dataset[0]
        )
        with torch.no_grad():
            pred, truth, sigmas = self._collect_predictions(loader)
        return self._metrics_from_arrays(pred, truth, title, result_key, sigmas)

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        dataset_titles = [
            "combined" if len(self.cfg.data.dataset) > 1 else self.cfg.data.dataset[0]
        ]
        model_core = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model_title = MODEL_TITLE_DICT[type(model_core.net).__name__]   
        #model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = [f"{model_title}: {dataset_title}" for dataset_title in dataset_titles]
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_test"]     = self.results_test
            plot_dict["results_train"]    = self.results_train
            plot_dict["results_per_proc"] = getattr(self, "results_per_proc", {})
        if self.cfg.train:
            plot_dict["train_loss"]        = self.train_loss
            plot_dict["val_loss"]          = self.val_loss
            plot_dict["train_lr"]          = self.train_lr
            plot_dict["train_loss_no_reg"] = getattr(self, "train_loss_no_reg", [])
            plot_dict["val_loss_no_reg"]   = getattr(self, "val_loss_no_reg",   [])
            plot_dict["train_mse"]         = getattr(self, "train_mse",         [])
            plot_dict["val_mse"]           = getattr(self, "val_mse",           [])
            plot_dict["proc_val_losses"]   = getattr(self, "proc_val_losses",   {})

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
        
    
        if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer"):
            return self._batch_loss_lloca(data)
        x, y, tokens = data   # (B, nparticles_max[,4]), (B,1), (B, nparticles_max)
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

    def _batch_loss_lloca(self, data):
        particles, y, tokens, ptr = data
        particles = particles.to(self.device)
        y         = y.to(self.device)
        tokens    = tokens.to(self.device)
        ptr       = ptr.to(self.device)
    
        y_pred = self.model(
            particles, tokens,
            mean = self.mom_mean[0],
            std  = self.mom_std[0],
            ptr  = ptr,
        )  # (B, out_channels)
    
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

    def _validate(self, step):
        """Override base _validate.

        Single dataset: delegates to base (no change).
        Multiple datasets: runs per-process val loaders only, derives combined
        loss as a simple mean across processes — consistent with the balanced
        sampler which gives each process equal batch slots during training.
        """
        if not self.proc_val_loaders:
            return super()._validate(step)

        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()

        proc_losses        = {}
        proc_losses_no_reg = {}
        proc_mse_vals      = {}

        with torch.no_grad():
            for name, loader in self.proc_val_loaders.items():
                losses, losses_no_reg, mse_vals = [], [], []
                for data in loader:
                    if self.ema is not None:
                        with self.ema.average_parameters():
                            loss, loss_no_reg, mse_val = self._batch_loss(data)
                    else:
                        loss, loss_no_reg, mse_val = self._batch_loss(data)
                    losses.append(loss.cpu().item())
                    if loss_no_reg is not None:
                        losses_no_reg.append(loss_no_reg)
                    if mse_val is not None:
                        mse_vals.append(mse_val)
                proc_losses[name]        = float(np.mean(losses))
                proc_losses_no_reg[name] = float(np.mean(losses_no_reg)) if losses_no_reg else None
                proc_mse_vals[name]      = float(np.mean(mse_vals))      if mse_vals      else None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Equal-weight combined loss across processes — matches the balanced sampler
        # which gives each process equal batch slots during training.
        val_loss = float(np.mean(list(proc_losses.values())))

        if (step + 1) % self.cfg.training.validate_every_n_steps == 0:
            self.val_loss.append(val_loss)
            lnr = [v for v in proc_losses_no_reg.values() if v is not None]
            if lnr:
                self.val_loss_no_reg.append(float(np.mean(lnr)))
            mse = [v for v in proc_mse_vals.values() if v is not None]
            if mse:
                self.val_mse.append(float(np.mean(mse)))
            #if self.cfg.use_mlflow:
            #    log_mlflow("val.loss", val_loss, step=step)

        # Record per-process losses for plotting
        for name, loss in proc_losses.items():
            if name not in self.proc_val_losses:
                self.proc_val_losses[name] = []
            self.proc_val_losses[name].append(loss)

        LOGGER.info(
            f"Val loss (combined): {val_loss:.4f} | " +
            ", ".join(f"{n}={v:.4f}" for n, v in proc_losses.items())
        )

        # Update sampler weights proportional to per-process loss
        if self.train_sampler is not None:
            weights = [proc_losses.get(name, 1.0) for name in self.cfg.data.dataset]
            self.train_sampler.set_weights(weights)
            LOGGER.info(
                "Sampler weights: " +
                ", ".join(f"{n}={w:.4f}" for n, w in zip(self.cfg.data.dataset, weights))
            )

        return val_loss
    
