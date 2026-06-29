import math

import numpy as np
import torch

import os
import json
import time
from omegaconf import OmegaConf, open_dict

from base_experiment import BaseExperiment
from dataset import AmplitudeDataset
from preprocessing import (
    preprocess_particles,
    preprocess_amplitude,
    undo_preprocess_amplitude,
    resolve_amp_trafos,
)
from plots import plot_mixer, short_ds_name
from logger import LOGGER
#from mlflow_util import log_mlflow
from losses import LogCoshLoss, RelL1Loss, HeteroscedasticLoss
from dataset import AmplitudeDataset, collate_variable_length

from lloca.utils.rand_transforms import rand_lorentz
from lloca.utils.polar_decomposition import restframe_boost

import psutil, os

# Coupling-order convention (see config/amplitudes.yaml): [n_loops, alpha_s_power].
#   LO=[0,0]  virt_only=[1,0]  NLO_full=[1,1]  NNLO=[2,2]
# Name-keyed so a new NLO/NNLO dataset file is labelled correctly without
# hand-editing positional amp_orders lists in every config. Checked most-specific
# token first; tree-level datasets (no nlo/nnlo token) fall back to LO.
_AMP_ORDER_BY_NAME = [
    ("nnlo",     [2, 2]),
    ("nlo_full", [1, 1]),
    ("nlo_virt", [1, 0]),   # absolute-virt (e4) and the virt/born ratio
]

def amp_order_for_dataset(name):
    """Map a dataset name to its coupling-order vector [n_loops, alpha_s_power]."""
    low = str(name).lower()
    for token, order in _AMP_ORDER_BY_NAME:
        if token in low:
            return list(order)
    return [0, 0]

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
    "LLOCAMuPTransformer": "LLoCa muP Transformer",
    "MuPLGATr": "L-GATr muP",
    "MuPLGATrSlim": "L-GATr-slim muP"
}


class AmplitudeExperiment(BaseExperiment):
    def init_physics(self):
        assert not self.cfg.training.force_xformers

        # Recipe-based data path: derive the dataset name list / amp_orders from
        # the `data.processes` spec before anything reads cfg.data.dataset.
        if self.cfg.data.get("source", "files") == "recipes":
            self._resolve_recipe_config()

        self.n_datasets = len(self.cfg.data.dataset)

        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

        if self.modelname in ["GAP", "MLP", "DSI"]:
            assert len(self.cfg.data.dataset) == 1, (
                f"Architecture {self.modelname} cannot handle several datasets"
            )

        # initialise tokenizer — will be populated in init_data
        from particle_ids import ParticleTokenizer
        tokenizer_path = os.path.join(self.cfg.run_dir, "particle_tokenizer.json")
        ft = self.cfg.get("fine_tune", None)
        ft_path = ft.get("pretrained_path", None) if ft is not None else None
        if self.warm_start and os.path.exists(tokenizer_path):
            self.tokenizer = ParticleTokenizer.load(tokenizer_path)
            LOGGER.info(f"Loaded tokenizer with vocab_size={self.tokenizer.vocab_size}")
        elif ft_path is not None:
            # Load tokenizer from pretrained model's run directory (two levels up from .pt)
            pretrained_run_dir = os.path.dirname(os.path.dirname(ft_path))
            pretrained_tok_path = os.path.join(pretrained_run_dir, "particle_tokenizer.json")
            if os.path.exists(pretrained_tok_path):
                self.tokenizer = ParticleTokenizer.load(pretrained_tok_path)
                LOGGER.info(
                    f"Fine-tuning: loaded tokenizer from pretrained run at {pretrained_tok_path} "
                    f"(vocab_size={self.tokenizer.vocab_size})"
                )
            else:
                LOGGER.warning(
                    f"Fine-tuning: no tokenizer found at {pretrained_tok_path}, using fresh tokenizer"
                )
                self.tokenizer = ParticleTokenizer()
        else:
            self.tokenizer = ParticleTokenizer()

        # mom_mean and mom_std are per-dataset, populated in init_data
        self.mom_mean = []
        self.mom_std  = []
        # Global physical momentum scale (recipe path: one train-fit std applied to
        # every process). The model sees momenta divided by it; needed to recover
        # physical GeV for data.mass_from_momenta. None until set in init_data.
        self.mom_div  = None

        if self.modelname not in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"):
            self.type_token = []
            for dataset in self.cfg.data.dataset:
                if self.cfg.data.include_permsym:
                    self.type_token.append(TYPE_TOKEN_DICT[dataset])
                else:
                    self.type_token.append(list(range(len(TYPE_TOKEN_DICT[dataset]))))
        
    def _resolve_recipe_config(self):
        """Parse the `data.processes` recipe spec into the per-process spec list
        and populate the config keys the rest of the pipeline expects
        (`data.dataset`, `data.amp_orders`). Runs once, at the top of
        init_physics, before n_datasets is read."""
        import mg5_pipeline_final as mg

        procs = self.cfg.data.get("processes", None)
        # Fall back to an external spec file (data.processes_file) — easier to
        # thread through a sweep than a nested list-of-dicts CLI override. The
        # file is a YAML/JSON list of the same per-process dicts.
        if not procs:
            pf = self.cfg.data.get("processes_file", None)
            if pf:
                import yaml
                with open(pf) as f:
                    procs = yaml.safe_load(f)
                # Allow either a bare list or {processes: [...]}.
                if isinstance(procs, dict):
                    procs = procs.get("processes", procs)
                LOGGER.info(f"Loaded {len(procs)} recipe processes from {pf}")
        assert procs, ("data.source=recipes requires data.processes or "
                       "data.processes_file to be set")

        specs, names, amp_orders = [], [], []
        for p in procs:
            p = OmegaConf.to_container(p, resolve=True) if OmegaConf.is_config(p) else dict(p)
            name = p["name"]
            sqrts = p["sqrts"]
            specs.append({
                "name":      name,
                "sqrts_min": float(sqrts[0]),
                "sqrts_max": float(sqrts[1]),
                "n_train":   int(p["n_train"]),
                "n_val":     int(p["n_val"]),
                "n_test":    int(p["n_test"]),
            })
            names.append(name)
            # Coupling order from the process definition (LO=[0,0]); explicit
            # override on the spec wins if given.
            if "amp_orders" in p and p["amp_orders"] is not None:
                amp_orders.append(list(p["amp_orders"]))
            else:
                k = mg.PROCESSES[name].get("alphas_power", 0)
                amp_orders.append([0, int(k)])

        self._recipe_specs = specs
        with open_dict(self.cfg):
            self.cfg.data.dataset    = names
            self.cfg.data.amp_orders = amp_orders
        LOGGER.info(
            f"Recipe data source: {len(names)} processes {names} "
            f"(seed={self.cfg.data.get('seed', 42)})"
        )

    def _boost_augment_momenta(self, particles, n_particles):
        """COM-boost + random-Lorentz augment a raw momentum block (the
        equivariant `prepare: lorentz` step). Returns the boosted/augmented
        momenta as a numpy array (N, n_particles, 4); the global `/std`
        standardization is applied separately so it can use one train-only scale
        across all processes (see _init_data_recipes)."""
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
        # Boost math is done in float64; store float32 — the model trains in
        # float32 and this halves the in-memory momentum footprint (matters at
        # the multi-million-event scale of the recipe pools).
        return particles_t.numpy().astype(np.float32)

    def _resolve_amp_orders(self, names):
        """Per-dataset [n_loops, alpha_s_power]. An explicit, length-matched
        data.amp_orders wins (e.g. the recipe path sets it per process); otherwise
        derive from dataset names so NLO/NNLO targets are labelled correctly instead
        of silently inheriting the positional LO default."""
        cfg_orders = self.cfg.data.get("amp_orders", None)
        if cfg_orders is not None and len(cfg_orders) == len(names):
            return [list(o) for o in cfg_orders]
        derived = [amp_order_for_dataset(n) for n in names]
        if cfg_orders is not None:
            LOGGER.warning(
                f"data.amp_orders has {len(cfg_orders)} entries but {len(names)} "
                f"dataset(s); deriving from names instead: {list(zip(names, derived))}")
        else:
            LOGGER.info(f"Derived amp_orders from dataset names: {list(zip(names, derived))}")
        with open_dict(self.cfg):
            self.cfg.data.amp_orders = derived
        return derived

    def init_data(self):
        if self.cfg.data.get("source", "files") == "recipes":
            return self._init_data_recipes()

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
        if subsample_cfg is None or (isinstance(subsample_cfg, str) and subsample_cfg.lower() in ("none", "null")):
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
    
            use_pids = self.cfg.data.get("use_PIDs", False)
            if use_pids:
                type_tokens = self.tokenizer.register_and_encode(pdg_ids)
            else:
                # Global fixed encoding: PDG ID → property-table index.
                # Independent of the training dataset, so a new BSM particle added
                # to PARTICLE_PROPERTIES in particle_ids.py is immediately usable
                # at inference without any model changes.
                from particle_ids import global_encode
                type_tokens = global_encode(pdg_ids)
                self.tokenizer.register_and_encode(pdg_ids)  # keep tokenizer in sync for use_PIDs=True warm starts

            # Raw per-slot PDG order (fixed per process) for the Tier-B leg→slot map.
            if not hasattr(self, "_slot_pdgs_by_name"):
                self._slot_pdgs_by_name = {}
            self._slot_pdgs_by_name.setdefault(dataset, [int(x) for x in pdg_ids[0]])

            if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"):
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
    
                _mom_div_ds = float(particles_t.std())
                particles_prepd = particles_t / _mom_div_ds
                self.mom_mean.append(float(particles_prepd.mean()))
                self.mom_std.append(float(particles_prepd.std().clamp(min=1e-2)))
                # Single-dataset legacy path has a well-defined global scale; with
                # several datasets each is scaled independently so there is no single
                # mom_div (mass_from_momenta is then unsupported — asserted at setup).
                self.mom_div = _mom_div_ds if self.n_datasets == 1 else None
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
        is_lloca = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim")
    
        all_particles_list = []
        all_tokens_list    = []
        all_amplitudes_raw = []
        all_process_ids    = []   # integer process index per event

        # amp_orders: list of [n_loops, alpha_s_power] per dataset, name-keyed
        # (NLO/NNLO targets labelled from their dataset name; LO otherwise).
        amp_orders = self._resolve_amp_orders(self.cfg.data.dataset)
        n_order_features = len(amp_orders[0])

        all_order_labels = []   # will be (N_events, n_order_features)

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
            order_row = np.array(amp_orders[proc_idx], dtype=np.float32)
            all_order_labels.append(np.tile(order_row, (N, 1)))  # (N, n_order_features)

        all_order_labels = np.concatenate(all_order_labels, axis=0)  # (N_total, n_order_features)

        all_amplitudes_raw = np.concatenate(all_amplitudes_raw, axis=0)  # (N_total, 1)
        all_process_ids    = np.array(all_process_ids, dtype=np.int32)   # (N_total,)
    
        # ------------------------------------------------------------------
        # Global amplitude preprocessing across all datasets combined
        # ------------------------------------------------------------------
        # Swap 'log' -> 'signedlog' when amplitudes contain non-positive values
        # (e.g. virtual corrections / virt-born ratios). Store the resolved list back
        # on the config so the inverse (undo_preprocess_amplitude) uses the same one.
        amp_trafos = resolve_amp_trafos(self.cfg.data.amp_trafos, all_amplitudes_raw)
        self.cfg.data.amp_trafos = amp_trafos
        LOGGER.info(
            f"Preprocessing amplitudes globally using trafos={amp_trafos}"
        )
        all_amplitudes_prepd, prepd_mean, prepd_std = preprocess_amplitude(
            all_amplitudes_raw, trafos=amp_trafos
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
        all_order_labels     = all_order_labels[perm]
    
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
        self.all_order_labels  = all_order_labels     # (N_events, n_order_features)
        self.n_order_features  = n_order_features
        self.N_events          = N_total

        self._finalize_data_sizing()

    def _finalize_data_sizing(self):
        """Shared tail of data init (legacy + recipe paths): derive the particle
        feature / order dimensions from the populated tokenizer + flat arrays and
        write the model in_channels, then save the tokenizer."""
        is_lloca = self.modelname in (
            "LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"
        )
        LOGGER.info(
            f"Combined dataset: {self.N_events} events, "
            f"{self.particles_flat.shape[0]} total particles"
        )

        # ------------------------------------------------------------------
        # Set model size — particle feature dimension depends on encoding mode
        # ------------------------------------------------------------------
        token_size = self.tokenizer.vocab_size
        self.token_size = token_size
        with open_dict(self.cfg):
            self.cfg.model.token_size = token_size

        if is_lloca:
            use_pids = self.cfg.data.get("use_PIDs", False)
            if use_pids:
                particle_feature_dim = token_size
                self.property_matrix = None
                LOGGER.info(f"Particle encoding: one-hot PIDs (vocab_size={token_size})")
            else:
                from particle_ids import (
                    ParticleFeaturizer, GLOBAL_N_ENTRIES, build_property_matrix,
                )
                # Smart-encoding transforms of the physical property vector, each
                # off by default so warm-start/fine-tune from existing checkpoints
                # stay valid (they change the encoder's *input* width — but not
                # n_scalars, which is fixed at d_particle_hidden + n_order_features):
                #   spin_onehot       — one-hot the categorical spin column
                #   color_onehot      — one-hot the SU(3) color representation
                #   prop_is_massless  — explicit "exactly massless" flag + neutralise sentinel
                #   standardize_props — z-score the continuous columns
                spin_onehot   = self.cfg.data.get("spin_onehot", False)
                color_onehot  = self.cfg.data.get("color_onehot", False)
                is_massless   = self.cfg.data.get("prop_is_massless", False)
                standardize   = self.cfg.data.get("standardize_props", False)
                self.property_matrix, _ = build_property_matrix(
                    spin_onehot=spin_onehot, color_onehot=color_onehot,
                    is_massless=is_massless, standardize=standardize,
                )
                # d_particle_hidden is the fixed projection output dim.
                # in_channels and num_scalars use this fixed dim — they never
                # change when quantum numbers are added, only the tiny projection
                # matrix (n_features → d_particle_hidden) needs extending.
                d_hidden = self.cfg.model.get("d_particle_hidden", 16)
                particle_feature_dim = d_hidden
                _enc = "+".join(
                    [n for n, on in (("spin1hot", spin_onehot),
                                     ("color1hot", color_onehot),
                                     ("masslessflag", is_massless),
                                     ("std", standardize)) if on]
                ) or "raw"
                LOGGER.info(
                    f"Particle encoding: physical properties "
                    f"({self.property_matrix.shape[1]}D [{_enc}] "
                    f"→ projected to {d_hidden}D) | "
                    f"global table covers {GLOBAL_N_ENTRIES - 1} particle species"
                )
            # n_order_features extra scalars per particle (same value broadcast across event)
            n_scalars   = particle_feature_dim + self.n_order_features

            # Feynman-diagram conditioning: load each process's diagram graphs and
            # widen the scalar channel by d_diag (the appended diagram embedding).
            # Requires the property encoding (use_PIDs=false). Off → no change.
            self._use_diagrams = (
                not use_pids and bool(self.cfg.model.get("use_diagrams", False))
            )
            if self._use_diagrams:
                self._d_diag = int(self.cfg.model.get("d_diag", 32))
                self._use_diag_virt = bool(self.cfg.model.get("use_diagram_virtuality", False))
                self._setup_diagram_registry(
                    list(self.cfg.data.dataset),
                    spin_onehot=spin_onehot, color_onehot=color_onehot,
                    is_massless=is_massless, standardize=standardize,
                    build_virtuality=self._use_diag_virt,
                    couplings_by_pid=getattr(self, "_coupling_by_pid", None),
                )
                n_scalars += self._d_diag
                LOGGER.info(
                    f"Diagram conditioning: ON (d_diag={self._d_diag}); "
                    f"n_scalars {n_scalars - self._d_diag}→{n_scalars}"
                )

            with open_dict(self.cfg):
                if self.modelname in ("MuPLGATr", "MuPLGATrSlim"):
                    # GATr nets: the 4-momentum is the (multi)vector input (in_*v_channels=1
                    # in the config), the particle/order encoding are the scalar inputs.
                    self.cfg.model.net.in_s_channels = n_scalars
                else:
                    self.cfg.model.net.in_channels = n_scalars + 4
                    self.cfg.model.net.num_scalars = n_scalars
            LOGGER.info(
                f"Order encoding: {self.n_order_features} features "
                f"(e.g. [n_loops, alpha_s_power]) → n_scalars={n_scalars}"
            )
        else:
            with open_dict(self.cfg):
                if self.modelname == "LGATr":
                    self.cfg.model.net.in_s_channels = token_size

        if self.cfg.save:
            tokenizer_path = os.path.join(self.cfg.run_dir, "particle_tokenizer.json")
            self.tokenizer.save(tokenizer_path)
            LOGGER.info(f"Saved tokenizer with vocab_size={token_size}")
            
    def _resolve_diagram_path(self, diagrams_dir, name):
        """Locate a process's diagram sidecar, tolerating dataset-name decoration.

        Recipe-mode names are registry keys (``ee_ttbar``) that match sidecar files
        directly; file-stem names (``ee_ttbar_346-1000GeV_amplitudes``) are retried
        after stripping a trailing ``_amplitudes`` and an energy tag. Returns the
        path or None."""
        import re
        cands = [name]
        base = re.sub(r"_amplitudes$", "", name)
        base = re.sub(r"_\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?GeV$", "", base)
        if base not in cands:
            cands.append(base)
        for c in cands:
            p = os.path.join(diagrams_dir, f"{c}.diagrams.json")
            if os.path.exists(p):
                return p
        return None

    def _setup_diagram_registry(self, names, spin_onehot, color_onehot,
                                is_massless, standardize, build_virtuality=False,
                                couplings_by_pid=None):
        """Load each process's diagram graphs into ``self._diag_pd_by_pid`` (indexed
        by process_id == position in ``data.dataset``). The property matrix uses the
        SAME smart-encoding flags as the particle encoder, so diagram particles ride
        the identical physical-quantity encoding. Missing sidecars → None (that
        process gets a zero diagram embedding; the run still works).

        ``couplings_by_pid`` : list[{order_key: alpha} | None] aligned with ``names``.
        When any entry is set, every vertex gets per-vertex log coupling-factor
        columns from that dataset's coupling values (see build_process_diagrams).
        Datasets without couplings are still given the columns (filled 0) so F_node
        stays uniform across the batched encoder forward."""
        from diagram_graphs import build_process_diagrams, feature_dims
        from particle_ids import build_property_matrix

        diagrams_dir = self.cfg.model.get("diagrams_dir", "data/diagrams")
        if not os.path.isabs(diagrams_dir):
            diagrams_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), diagrams_dir)
        enc_cfg = self.cfg.model.get("diagram_encoder", {}) or {}
        k_pe = int(enc_cfg.get("k_pe", 8))
        max_diagrams = enc_cfg.get("max_diagrams", None)
        max_diagrams = int(max_diagrams) if max_diagrams is not None else None

        prop_matrix, _ = build_property_matrix(
            spin_onehot=spin_onehot, color_onehot=color_onehot,
            is_massless=is_massless, standardize=standardize,
        )
        # Couplings on vertices: if ANY dataset carries coupling values, every
        # process gets the coupling columns (0-filled where absent) so the batched
        # encoder sees one uniform F_node.
        use_couplings = bool(couplings_by_pid) and any(c for c in couplings_by_pid)
        empty_coupl = {} if use_couplings else None   # {} → columns present, all 0

        pd_by_pid, missing = [], []
        for i, name in enumerate(names):
            path = self._resolve_diagram_path(diagrams_dir, name)
            if path is None:
                pd_by_pid.append(None)
                missing.append(name)
                continue
            cpl = (couplings_by_pid[i] if couplings_by_pid else None) or empty_coupl
            pd_by_pid.append(build_process_diagrams(
                path, prop_matrix, k_pe=k_pe, max_diagrams=max_diagrams,
                couplings=cpl))

        self._diag_pd_by_pid = pd_by_pid
        self._diag_feature_dims = feature_dims(
            prop_matrix.shape[1], with_couplings=use_couplings)   # (f_node, f_edge)
        self._diag_k_pe = k_pe

        # Tier B: per-process propagator-virtuality precompute. Needs the event's
        # per-slot PDG order (recover raw PDG from the stored property-table indices
        # via the inverse global map) and the number of initial-state legs.
        self._diag_virt_by_pid = None
        if build_virtuality:
            from diagram_graphs import build_process_virtuality
            slot_map = getattr(self, "_slot_pdgs_by_name", {})            # set in both data paths
            virt, n_virt = [], 0
            for i, pd in enumerate(pd_by_pid):
                slot_pdgs = slot_map.get(names[i])
                if pd is None or slot_pdgs is None:
                    virt.append(None); continue
                n_initial = sum(1 for leg in (pd.external or []) if leg["state"] == "in")
                vt = build_process_virtuality(pd, [int(x) for x in slot_pdgs], n_initial)
                virt.append(vt)
                n_virt += int(vt is not None)
            self._diag_virt_by_pid = virt
            LOGGER.info(f"Diagram virtuality (Tier B): built per-event maps for "
                        f"{n_virt}/{len(pd_by_pid)} processes")
        if missing:
            LOGGER.warning(
                f"Diagram conditioning: no sidecar for {missing} (zero embedding "
                f"used). Generate via `python tools/dump_diagrams.py --all`.")
        n_ok = sum(p is not None for p in pd_by_pid)
        LOGGER.info(
            f"Diagram conditioning: loaded graphs for {n_ok}/{len(names)} processes "
            f"from {diagrams_dir} (f_node={self._diag_feature_dims[0]}, "
            f"f_edge={self._diag_feature_dims[1]}, k_pe={k_pe})")

    def _post_instantiate_model(self, model):
        """Called on the instantiated model after construction.
        Adds particle_encoder. Since the μP backbone computes its own base shapes in
        __init__, particle_encoder is added outside that scope and is later marked as a
        standard-parametrization (SP) parameter by lloca.mup.finalize (in init_model)."""
        if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"):
            use_pids = self.cfg.data.get("use_PIDs", False)
            model.setup_particle_features(
                use_pids=use_pids,
                property_matrix=getattr(self, "property_matrix", None),
                encoder_hidden=self.cfg.model.get("particle_encoder_hidden", 0),
            )
            # Data-derived per-particle on-shell mass (replaces the frozen
            # log10_mass_gev table column with m=sqrt(E^2-|p|^2) per particle).
            if self.cfg.data.get("mass_from_momenta", False):
                assert not use_pids, (
                    "data.mass_from_momenta requires the property encoding "
                    "(use_PIDs=false)"
                )
                assert self.mom_div is not None, (
                    "data.mass_from_momenta needs a single global momentum scale; "
                    "use the recipe data path or a single-dataset run."
                )
                from particle_ids import mass_feature_spec
                spec = mass_feature_spec(
                    spin_onehot=self.cfg.data.get("spin_onehot", False),
                    color_onehot=self.cfg.data.get("color_onehot", False),
                    is_massless=self.cfg.data.get("prop_is_massless", False),
                    standardize=self.cfg.data.get("standardize_props", False),
                )
                model.setup_mass_from_momenta(self.mom_div, spec)
                LOGGER.info(
                    f"Mass encoding: data-derived on-shell mass (mom_div={self.mom_div:.4g}, "
                    f"mass_col={spec['mass_col']}, replaces frozen table mass)"
                )
            # Build + attach the diagram graph encoder (a real submodule → trained,
            # checkpointed, marked SP by mup_finalize). Done here, like particle_encoder,
            # so it exists before μP finalisation and warm-start loading.
            if getattr(self, "_use_diagrams", False):
                from models.diagram_encoder import DiagramEncoder
                f_node, f_edge = self._diag_feature_dims
                enc_cfg = self.cfg.model.get("diagram_encoder", {}) or {}
                # Tier B "edge" mode adds one per-event graph edge feature (the
                # virtuality); "pool" mode feeds virtuality only into the pooling, so
                # the graph encoder needs no extra edge channel.
                _virt_mode = str(self.cfg.model.get("virt_mode", "edge"))
                f_edge_extra = 1 if (getattr(self, "_use_diag_virt", False)
                                     and _virt_mode == "edge") else 0
                encoder = DiagramEncoder(
                    f_node=f_node, f_edge=f_edge, k_pe=self._diag_k_pe,
                    d_model=int(enc_cfg.get("d_model", 64)),
                    n_heads=int(enc_cfg.get("n_heads", 4)),
                    n_layers=int(enc_cfg.get("n_layers", 3)),
                    d_out=self._d_diag, f_edge_extra=f_edge_extra,
                )
                model.setup_diagram_conditioning(encoder, self._diag_pd_by_pid, self._d_diag)
                if getattr(self, "_use_diag_virt", False):
                    model.setup_diagram_virtuality(
                        self._diag_virt_by_pid,
                        log_scale=float(self.cfg.model.get("virt_log_scale", 0.1)),
                        standardize=bool(self.cfg.model.get("virt_standardize", True)),
                        clamp=float(self.cfg.model.get("virt_clamp", 4.0)),
                        mode=_virt_mode)

    def init_model(self):
        super().init_model()  # _post_instantiate_model is called inside for all three models

        ft = self.cfg.get("fine_tune", None)
        if ft is None or ft.get("pretrained_path", None) is None:
            return

        self._load_pretrained_weights(
            ft.pretrained_path,
            reset_output_head=ft.get("reset_output_head", False),
        )

        if ft.lora.get("enabled", False):
            from fine_tune import inject_lora
            inject_lora(
                self.model,
                rank=ft.lora.rank,
                alpha=ft.lora.alpha,
                target=list(ft.lora.target),
            )

        freeze_blocks = list(ft.get("freeze_blocks", []))
        if freeze_blocks:
            for idx in freeze_blocks:
                for p in self.model.net.net.blocks[idx].parameters():
                    p.requires_grad_(False)
            LOGGER.info(f"Fine-tuning: froze transformer blocks {freeze_blocks}")

    def _init_ewc(self):
        ft = self.cfg.get("fine_tune", None)
        if ft is None or not ft.ewc.get("enabled", False):
            self.ewc = None
            return
        from fine_tune import EWC
        loss_fn = lambda batch: self._batch_loss(batch)[0]
        self.ewc = EWC(
            self.model,
            self.train_loader,
            n_fisher_batches=ft.ewc.get("n_fisher_batches", 64),
            device=self.device,
            loss_fn=loss_fn,
        )
        LOGGER.info(f"Fine-tuning: EWC initialized (lambda={ft.ewc.get('lambda', 1000.0)})")

    def _init_data_recipes(self):
        """Recipe-based data path: materialize explicit per-role pools
        (train / frozen val / frozen test) from the `data.processes` spec, build
        the combined flat arrays, and normalize using train-only statistics that
        are frozen and saved to the run dir (reloaded on warm start rather than
        recomputed)."""
        import datagen
        from particle_ids import global_encode
        from dataset import build_flat_arrays

        assert self.modelname in (
            "LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"
        ), "data.source=recipes is only supported for the LLoCa μP model"

        specs     = self._recipe_specs
        names     = [s["name"] for s in specs]
        seed      = int(self.cfg.data.get("seed", 42))
        use_pids  = self.cfg.data.get("use_PIDs", False)
        roles     = ("train", "val", "test")
        count_key = {"train": "n_train", "val": "n_val", "test": "n_test"}

        # --- materialize the three role pools (frozen val/test, cached train) ---
        # require_cache (default true on the recipe path) means a GPU training job
        # never generates data inline — it expects the prebuild job
        # (prebuild_recipes.sh) to have populated the cache, and fails fast
        # otherwise rather than burning GPU time. Set data.require_cache=false to
        # allow inline generation (e.g. a quick local run).
        require_cache = bool(self.cfg.data.get("require_cache", True))
        paths = {}
        for role in roles:
            role_specs = [{
                "process":   s["name"],
                "sqrts_min": s["sqrts_min"],
                "sqrts_max": s["sqrts_max"],
                "n_events":  s[count_key[role]],
            } for s in specs]
            paths[role] = datagen.ensure_split_set(
                role_specs, role=role, seed=seed, require_cache=require_cache)

        # --- frozen normalization stats (all GLOBAL scalars, train-only) ---
        # Source of truth, in priority order:
        #   1. warm-start resume  -> this run's own data_stats.json
        #   2. fine-tune          -> the pretrained run's data_stats.json, so the
        #                            backbone keeps seeing inputs in its training
        #                            normalization (mirrors the tokenizer reuse)
        #   3. otherwise          -> compute fresh from train below
        stats_path = os.path.join(self.cfg.run_dir, "data_stats.json")
        ft         = self.cfg.get("fine_tune", None)
        ft_path    = ft.get("pretrained_path", None) if ft is not None else None
        stats_src  = None
        if self.warm_start and os.path.exists(stats_path):
            stats_src = stats_path
        elif ft_path is not None:
            cand = os.path.join(os.path.dirname(os.path.dirname(ft_path)),
                                "data_stats.json")
            if os.path.exists(cand):
                stats_src = cand

        # Amplitude preprocessing scope: global (single train-only scale, keeps
        # the relative magnitude differences between processes) vs per-dataset
        # (each process normalized to unit scale independently). Controlled by
        # data.preprocess_per_dataset (default False = global).
        per_dataset = bool(self.cfg.data.get("preprocess_per_dataset", False))

        if stats_src is not None:
            with open(stats_src) as f:
                stats = json.load(f)
            mom_div    = float(stats["mom_div"])
            mom_mean   = float(stats["mom_mean"])
            mom_std    = float(stats["mom_std"])
            amp_trafos = stats["amp_trafos"]
            # prepd_mean/std stored as a list: length 1 (global) or n_proc
            # (per-dataset, aligned with data.dataset order).
            prepd_means = [float(x) for x in np.atleast_1d(np.asarray(stats["prepd_mean"], dtype=np.float64))]
            prepd_stds  = [float(x) for x in np.atleast_1d(np.asarray(stats["prepd_std"],  dtype=np.float64))]
            per_dataset = bool(stats.get("preprocess_per_dataset", len(prepd_means) > 1))
            LOGGER.info(f"Loaded frozen data stats from {stats_src}")
        else:
            mom_div = mom_mean = mom_std = None
            amp_trafos = None
            prepd_means = prepd_stds = None

        amp_orders       = self._resolve_amp_orders(self.cfg.data.dataset)
        n_order_features = len(amp_orders[0])

        # Per-dataset load caps: the frozen pools can hold far more than a single
        # run trains on (e.g. 10M/process), so only `train_subsample` train events
        # and `eval_subsample` val/test events per process are pulled into memory
        # (mmap slice — the full file is never read). None = load all.
        train_cap = self.cfg.data.get("train_subsample", None)
        eval_cap  = self.cfg.data.get("eval_subsample", None)
        cap_for   = {"train": train_cap, "val": eval_cap, "test": eval_cap}

        # --- load every (role, process); boost+augment momenta (no scale yet) ---
        store = {}
        for role in roles:
            cap = cap_for[role]
            cap = int(cap) if cap not in (None, "null", "none") else None
            for s in specs:
                name = s["name"]
                if cap is not None:
                    data_raw = np.asarray(np.load(paths[role][name], mmap_mode="r")[:cap])
                else:
                    data_raw = np.load(paths[role][name])
                LOGGER.info(
                    f"  [{role}] {name}: {data_raw.shape[0]} events"
                    f"{f' (cap {cap})' if cap else ''} from {paths[role][name]}"
                )
                n_particles  = (data_raw.shape[1] - 1) // 5
                momenta_cols = n_particles * 4
                particles  = data_raw[:, :momenta_cols]
                pdg_ids    = data_raw[:, momenta_cols:-1].astype(int)
                amplitudes = data_raw[:, [-1]]

                if use_pids:
                    toks = self.tokenizer.register_and_encode(pdg_ids)
                else:
                    toks = global_encode(pdg_ids)
                    self.tokenizer.register_and_encode(pdg_ids)  # keep in sync

                # Raw per-slot PDG order (fixed per process) for the Tier-B leg→slot map.
                if not hasattr(self, "_slot_pdgs_by_name"):
                    self._slot_pdgs_by_name = {}
                self._slot_pdgs_by_name.setdefault(name, [int(x) for x in pdg_ids[0]])

                store[(role, name)] = {
                    "momenta": self._boost_augment_momenta(particles, n_particles),
                    "tokens":  toks,
                    "raw_amp": amplitudes,
                }

        # --- single GLOBAL momentum scale, fit on TRAIN only (matches upstream
        #     load_file: momentum /= momentum.std()), then applied to every block.
        #     mom_mean/mom_std for the in-net (local-mean)/std are the global
        #     stats of the scaled train momenta. ---
        if mom_div is None:
            train_mom = np.concatenate(
                [store[("train", n)]["momenta"].reshape(-1, 4) for n in names], axis=0
            )
            mom_div  = float(train_mom.std())
            scaled   = train_mom / mom_div
            mom_mean = float(scaled.mean())
            mom_std  = float(max(scaled.std(), 1e-2))
        self.mom_mean = [mom_mean]
        self.mom_std  = [mom_std]
        self.mom_div  = float(mom_div)   # physical scale (for data.mass_from_momenta)
        for k in store:
            store[k]["momenta"] = store[k]["momenta"] / mom_div

        # --- amplitude preprocessing: stats fitted on TRAIN, applied to all ---
        # The amp_trafos list (e.g. signedlog) is always resolved globally so the
        # same transform is used for every process; only the standardization
        # mean/std differ between the global and per-dataset scopes.
        train_raw = np.concatenate(
            [store[("train", n)]["raw_amp"] for n in names], axis=0
        )
        if amp_trafos is None:
            amp_trafos = resolve_amp_trafos(self.cfg.data.amp_trafos, train_raw)
        self.cfg.data.amp_trafos = amp_trafos
        if prepd_means is None:
            if per_dataset:
                prepd_means, prepd_stds = [], []
                for name in names:
                    _, m, s = preprocess_amplitude(
                        store[("train", name)]["raw_amp"], trafos=amp_trafos)
                    prepd_means.append(float(m)); prepd_stds.append(float(s))
            else:
                _, m, s = preprocess_amplitude(train_raw, trafos=amp_trafos)
                prepd_means, prepd_stds = [float(m)], [float(s)]
        self.prepd_mean = prepd_means
        self.prepd_std  = prepd_stds
        LOGGER.info(
            f"Preprocessing amplitudes "
            f"{'per-dataset (independent unit scale)' if per_dataset else 'globally (shared scale)'} "
            f"using trafos={amp_trafos} (train-only stats); "
            f"means={[f'{m:.3f}' for m in prepd_means]}"
        )

        # --- assemble combined flat arrays, role-contiguous (train|val|test) ---
        # Vectorized per dataset (no per-event Python list), so this scales to
        # millions of events. Within a (role, process) block every event has the
        # same particle count P, so the boosted momenta (N,P,4) reshape directly
        # to a flat (N*P,4) and offsets are built arithmetically. The flat
        # particle arrays stay in dataset order; only the small per-event arrays
        # (offsets / amp / pid / order) are shuffled (seed 42) to interleave
        # processes for the balanced sampler — the offsets still index the right
        # (unshuffled) particle ranges, so O(events) work touches only metadata.
        role_particles, role_tokens, role_offsets = [], [], []
        role_amp, role_pid, role_order = [], [], []
        role_counts = {}
        rng = np.random.default_rng(seed=42)
        part_base = 0   # cumulative particle offset into the final flat array

        for role in roles:
            ds_parts, ds_toks = [], []
            ev_starts, ev_pid, ev_order, amp_blocks = [], [], [], []
            P_of = []
            for proc_idx, name in enumerate(names):
                rec   = store[(role, name)]
                parts = rec["momenta"]          # (N, P, 4)
                toks  = rec["tokens"]           # (N, P)
                N, P  = parts.shape[0], parts.shape[1]
                m = prepd_means[proc_idx] if per_dataset else prepd_means[0]
                s = prepd_stds[proc_idx]  if per_dataset else prepd_stds[0]
                amp_prepd, _, _ = preprocess_amplitude(
                    rec["raw_amp"], trafos=amp_trafos, mean=m, std=s,
                )
                ds_parts.append(parts.reshape(N * P, 4))
                ds_toks.append(np.asarray(toks).reshape(N * P))
                ev_starts.append(part_base + np.arange(N, dtype=np.int64) * P)
                P_of.append(np.full(N, P, dtype=np.int64))
                ev_pid.append(np.full(N, proc_idx, dtype=np.int32))
                ev_order.append(np.tile(
                    np.array(amp_orders[proc_idx], dtype=np.float32), (N, 1)))
                amp_blocks.append(amp_prepd)
                part_base += N * P

            starts = np.concatenate(ev_starts)
            ends   = starts + np.concatenate(P_of)
            offs   = np.stack([starts, ends], axis=1)        # (N_role, 2)
            amp    = np.concatenate(amp_blocks, axis=0)
            pid    = np.concatenate(ev_pid)
            order  = np.concatenate(ev_order, axis=0)

            perm = rng.permutation(offs.shape[0])
            role_particles.append(np.concatenate(ds_parts, axis=0))
            role_tokens.append(np.concatenate(ds_toks, axis=0))
            role_offsets.append(offs[perm])
            role_amp.append(amp[perm])
            role_pid.append(pid[perm])
            role_order.append(order[perm])
            role_counts[role] = offs.shape[0]

        # dtypes chosen so AmplitudeDataset (torch.as_tensor) shares these buffers
        # zero-copy across all loaders instead of copying per loader.
        self.particles_flat   = np.concatenate(role_particles, axis=0).astype(np.float32, copy=False)
        self.tokens_flat      = np.concatenate(role_tokens, axis=0).astype(np.int64, copy=False)
        self.offsets          = np.concatenate(role_offsets, axis=0)
        self.all_amplitudes   = np.concatenate(role_amp, axis=0).astype(np.float32, copy=False)
        self.all_process_ids  = np.concatenate(role_pid, axis=0).astype(np.int64, copy=False)
        self.all_order_labels = np.concatenate(role_order, axis=0).astype(np.float32, copy=False)
        self.n_order_features = n_order_features
        self.N_events         = self.offsets.shape[0]
        self._role_counts     = (role_counts["train"], role_counts["val"],
                                  role_counts["test"])

        # --- freeze stats to this run's dir (skip if already present, i.e. a
        #     warm-start resume; a fine-tune persists the inherited stats here so
        #     the fine-tune run can itself be resumed reproducibly). ---
        if self.cfg.save and not os.path.exists(stats_path):
            with open(stats_path, "w") as f:
                json.dump({
                    "mom_div":    float(mom_div),
                    "mom_mean":   float(self.mom_mean[0]),
                    "mom_std":    float(self.mom_std[0]),
                    "amp_trafos": list(amp_trafos) if amp_trafos else amp_trafos,
                    "preprocess_per_dataset": per_dataset,
                    "prepd_mean": [float(x) for x in self.prepd_mean],
                    "prepd_std":  [float(x) for x in self.prepd_std],
                }, f, indent=2)
            LOGGER.info(f"Saved frozen data stats to {stats_path}")

        self._finalize_data_sizing()

    def _init_dataloader(self):
        from dataset import AmplitudeDataset, collate_variable_length, ProcessBalancedSampler

        N_total = self.N_events

        role_counts = getattr(self, "_role_counts", None)
        if role_counts is not None:
            # Recipe data path: explicit, role-contiguous train|val|test pools
            # (no positional ratio split).
            n_train, n_val, n_test = role_counts
            train_idx = np.arange(0,                 n_train)
            val_idx   = np.arange(n_train,           n_train + n_val)
            test_idx  = np.arange(n_train + n_val,   n_train + n_val + n_test)
            with open_dict(self.cfg):
                self.cfg.data.subsample = n_train
        else:
            assert sum(self.cfg.data.train_test_val) <= 1

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
                order_labels   = self.all_order_labels[indices],
                process_ids    = self.all_process_ids[indices],
                dtype          = self.dtype,
            )

        def make_loader(indices, shuffle, batchsize, sampler=None, workers=None):
            # workers defaults to nw (cfg.training.num_workers); eval / per-process loaders
            # pass workers=0 so the large in-memory dataset isn't forked into many worker
            # sets (one per loader), whose copy-on-write pages creep the RSS up.
            #
            # pin_memory=False (NOT w>0): batches are variable-length (the collated particle
            # tensor size differs every step), so a pinned-memory caching allocator hoards a
            # distinct page-locked buffer per distinct size -> host RAM grows unbounded ->
            # OOM over a long run. This is independent of workers/persistence (it bit every
            # config). Dropping pin_memory removes the pinned cache; the H2D copy is then
            # sync, but it's tiny (~1 MB/batch) next to compute, and the worker collate
            # overlap (the real speedup) is unaffected. persistent_workers stays True so
            # workers are never respawned mid-run (respawning leaks semaphores/procs and
            # progressively slows the run).
            ds = make_dataset(indices)
            w = nw if workers is None else workers
            return torch.utils.data.DataLoader(
                ds,
                batch_size  = batchsize,
                shuffle     = shuffle if sampler is None else False,
                sampler     = sampler,
                drop_last   = True,
                collate_fn  = collate_variable_length,
                pin_memory         = False,
                num_workers        = w,
                persistent_workers = w > 0,
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
        self.train_eval_loader = make_loader(train_idx, shuffle=False, batchsize=train_eval_bs,
                                             workers=0)

        eval_bs = min(self.cfg.evaluation.batchsize, max(len(val_idx) // 2, 1))
        self.val_loader  = make_loader(val_idx,  shuffle=False, batchsize=eval_bs, workers=0)
        self.test_loader = make_loader(test_idx, shuffle=False,
                                       batchsize=min(self.cfg.evaluation.batchsize,
                                                     max(len(test_idx) // 2, 1)),
                                       workers=0)

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
                        batchsize=min(eval_bs, max(len(p_idx) // 2, 1)),
                        workers=0,
                    )
                    LOGGER.info(f"  Per-process val loader '{name}': {len(p_idx)} events")
                # test
                mask  = test_proc_ids == p
                p_idx = test_idx[mask]
                if len(p_idx) >= 2:
                    self.proc_test_loaders[name] = make_loader(
                        p_idx, shuffle=False,
                        batchsize=min(test_bs, max(len(p_idx) // 2, 1)),
                        workers=0,
                    )
                # train (plain, for evaluation only)
                mask  = train_proc_ids == p
                p_idx = train_idx[mask]
                if len(p_idx) >= 2:
                    self.proc_train_eval_loaders[name] = make_loader(
                        p_idx, shuffle=False,
                        batchsize=min(train_eval_bs, max(len(p_idx) // 2, 1)),
                        workers=0,
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

        # Per-process amplitude un-standardization stats. With per-dataset
        # preprocessing self.prepd_mean is aligned with data.dataset order; with
        # global preprocessing it is length-1 and every process shares stats[0].
        per_dataset_amp = len(self.prepd_mean) > 1
        def _amp_stats(name):
            if per_dataset_amp:
                p = list(self.cfg.data.dataset).index(name)
                return self.prepd_mean[p], self.prepd_std[p]
            return self.prepd_mean[0], self.prepd_std[0]

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
                pred, truth, sigmas = collect(loader)
                return pred, truth, sigmas, self.prepd_mean[0], self.prepd_std[0]
            pred   = np.concatenate([proc_preds[n][split][0] for n in available], axis=0)
            truth  = np.concatenate([proc_preds[n][split][1] for n in available], axis=0)
            first_sig = proc_preds[available[0]][split][2]
            sigmas = (np.concatenate([proc_preds[n][split][2] for n in available], axis=0)
                      if first_sig is not None else None)
            # Per-event un-standardization stats, in the same concat order, so the
            # combined raw-amplitude metrics undo each process correctly under
            # per-dataset preprocessing (no-op broadcast under global).
            pm = np.concatenate([
                np.full((proc_preds[n][split][1].shape[0], 1), _amp_stats(n)[0])
                for n in available], axis=0)
            ps = np.concatenate([
                np.full((proc_preds[n][split][1].shape[0], 1), _amp_stats(n)[1])
                for n in available], axis=0)
            return pred, truth, sigmas, pm, ps

        # ------------------------------------------------------------------
        # Compute metrics (pure numpy, fast)
        # ------------------------------------------------------------------
        LOGGER.info("### Computing combined metrics ###")
        for split, attr in [("train", "results_train"), ("val", "results_val"), ("test", "results_test")]:
            pred, truth, sigmas, pm, ps = concat_split(split)
            setattr(self, attr,
                    self._metrics_from_arrays(pred, truth, split, combined_key, sigmas,
                                              prepd_mean=pm, prepd_std=ps))

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
            pm, ps = _amp_stats(name)
            for split, (pred, truth, sigmas) in splits.items():
                self.results_per_proc[name][split] = self._metrics_from_arrays(
                    pred, truth, f"{split}_{name}", name, sigmas,
                    prepd_mean=pm, prepd_std=ps,
                )[name]

        # Optionally log noema metrics (no extra forward pass — just re-use arrays)
        if self.ema is not None:
            LOGGER.info("### Evaluating without EMA (reusing predictions not possible — skipping noema) ###")

        return self.results

    def call_model_fn(self, x, tokens=None):
        if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"):
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
        is_lloca  = self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim")
        all_pred  = []
        all_truth = []
        all_sigma = [] if self.cfg.training.loss == "HETEROSC" else None

        t0 = time.time()
        for data in loader:
            particles, y, tokens, order_labels, ptr, process_ids = data
            if is_lloca:
                particles    = particles.to(self.device)
                tokens       = tokens.to(self.device)
                order_labels = order_labels.to(self.device)
                ptr          = ptr.to(self.device)
                process_ids  = process_ids.to(self.device)
                y_pred = self.model(
                    particles, tokens,
                    mean=self.mom_mean[0], std=self.mom_std[0],
                    ptr=ptr, order_labels=order_labels,
                    process_ids=process_ids,
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
                             sigmas=None, prepd_mean=None, prepd_std=None):
        """Compute metrics from preprocessed arrays (no model call). Pure numpy.

        `prepd_mean`/`prepd_std` override the standardization stats used to undo
        the amplitude preprocessing. They may be scalars or per-event arrays
        (broadcastable to amp shape) — needed when amplitudes were preprocessed
        per-dataset, so each event is un-standardized with its own process's
        stats. Default to the global stats (self.prepd_mean[0])."""
        if prepd_mean is None:
            prepd_mean = self.prepd_mean[0]
        if prepd_std is None:
            prepd_std = self.prepd_std[0]
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
            amp_truth_prepd, prepd_mean, prepd_std,
            trafos=self.cfg.data.amp_trafos,
        )
        amp_pred = undo_preprocess_amplitude(
            amp_pred_prepd, prepd_mean, prepd_std,
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
            "combined" if len(self.cfg.data.dataset) > 1 else short_ds_name(self.cfg.data.dataset[0])
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
            plot_dict["proc_val_losses"]        = getattr(self, "proc_val_losses",        {})
            plot_dict["proc_val_losses_no_reg"] = getattr(self, "proc_val_losses_no_reg", {})
            plot_dict["proc_ema_losses"]        = getattr(self, "_proc_ema_losses",        {})
            plot_dict["proc_ema_losses_no_reg"] = getattr(self, "_proc_ema_losses_no_reg", {})
            plot_dict["validate_every_n_steps"] = self.cfg.training.validate_every_n_steps
            # per-process cumulative compute (samples seen) at each validation step;
            # list[dict{proc_id: n_samples}], aligned with proc_val_losses.
            plot_dict["proc_compute_snapshots"] = (
                self.train_sampler.compute_snapshots if self.train_sampler is not None else []
            )
            plot_dict["dataset_order"] = list(self.cfg.data.dataset)
            # solo scaling reference {dataset: [a, b]} for the vs-compute overlay
            dp = (self.cfg.data.get("data_path", "") or "")
            solo_path = os.path.join(dp, "solo_scaling.json") if dp else None
            try:
                if solo_path and os.path.exists(solo_path):
                    plot_dict["solo_scaling"] = json.load(open(solo_path))
            except Exception as e:
                LOGGER.warning(f"Could not load solo_scaling reference ({solo_path}): {e}")

        self._save_per_process_metrics(plot_path)
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _save_per_process_metrics(self, plot_path):
        """Dump per-process validation metrics to JSON for later reconstruction.

        Saves the raw per-dataset val losses (with/without reg), the EMA-smoothed
        series the sampler fits α on, the per-dataset cumulative compute, and the
        combined val losses — enough to rebuild the global loss, the EMA curves,
        and the loss-vs-compute scaling offline.
        """
        if not self.cfg.train or not getattr(self, "proc_val_losses", {}):
            return
        dataset_order = list(self.cfg.data.dataset)
        snapshots = (
            self.train_sampler.compute_snapshots if self.train_sampler is not None else []
        )
        proc_compute = {
            name: [int(s.get(i, 0)) for s in snapshots]
            for i, name in enumerate(dataset_order)
        }

        def _f(seq):
            return [float(v) for v in seq]

        def _fd(d):
            return {k: _f(v) for k, v in d.items()}

        dump = {
            "validate_every_n_steps": self.cfg.training.validate_every_n_steps,
            "dataset_order": dataset_order,
            "val_loss": _f(self.val_loss),
            "val_loss_no_reg": _f(getattr(self, "val_loss_no_reg", [])),
            "proc_val_losses": _fd(self.proc_val_losses),
            "proc_val_losses_no_reg": _fd(getattr(self, "proc_val_losses_no_reg", {})),
            "proc_ema_losses": _fd(getattr(self, "_proc_ema_losses", {})),
            "proc_ema_losses_no_reg": _fd(getattr(self, "_proc_ema_losses_no_reg", {})),
            "proc_compute": proc_compute,
            "proc_status": getattr(self, "_ds_status_hist", {}),
        }
        out_path = os.path.join(plot_path, "per_process_metrics.json")
        with open(out_path, "w") as f:
            json.dump(dump, f)
        LOGGER.info(f"Saved per-process metrics to {out_path}")

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
        # The naive `sum(p.pow(2).sum() for p in model.parameters())` launches one
        # pow + one reduction kernel (plus a full-size temporary) per parameter
        # tensor and chains the adds in Python — ~100+ tiny serialized kernels every
        # step for an 8-block transformer, independent of width. torch._foreach_norm
        # fuses all tensors into a handful of multi-tensor kernels (the same
        # machinery clip_grad_norm_ uses). Set LLOCA_REG=loop for the old path.
        use_loop = os.environ.get("LLOCA_REG", "foreach") == "loop"
        match self.cfg.training.regularization:
            case "L2":
                if use_loop:
                    self.regularization = lambda model: sum(param.pow(2.0).sum() for param in model.parameters())
                else:
                    # sum of squared L2 norms == sum of squares of all params.
                    # norm-then-square introduces a tiny float rounding difference
                    # vs. pow(2).sum() — negligible for a regularization term.
                    self.regularization = lambda model: torch.stack(
                        torch._foreach_norm(list(model.parameters()))
                    ).square().sum()
            case "L1":
                if use_loop:
                    self.regularization = lambda model: sum(param.abs().sum() for param in model.parameters())
                else:
                    # L1 norm per tensor (ord=1), summed → sum of |p| over all params (exact).
                    self.regularization = lambda model: torch.stack(
                        torch._foreach_norm(list(model.parameters()), ord=1)
                    ).sum()
            case None:
                self.regularization = lambda model: 0.0
            case _:
                raise ValueError(
                    f"Unknown regularization function {self.cfg.training.regularization}"
                )

    def _batch_loss(self, data):
        
    
        if self.modelname in ("LLOCATransformer", "LLOCAMuPTransformer", "MuPLGATr", "MuPLGATrSlim"):
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
        # Detached tensor (not .item()) — deferred sync; see _batch_loss_lloca / _step.
        loss_no_reg = loss.detach()
        loss        = loss + reg_term
        if os.environ.get("LLOCA_SYNC", "deferred") == "blocking":
            assert torch.isfinite(loss).all()
        return loss, loss_no_reg, mse_val

    def _batch_loss_lloca(self, data):
        particles, y, tokens, order_labels, ptr, process_ids = data

        # Compute the per-event particle counts on the CPU (ptr is born on the CPU
        # in collate_variable_length) so the attention-mask builder doesn't have to
        # `.tolist()` the GPU ptr — that sync drains the CUDA queue every forward.
        # Disabled under LLOCA_SYNC=blocking (original device-ptr path) for A/B.
        sync_blocking = os.environ.get("LLOCA_SYNC", "deferred") == "blocking"
        seq_lens = None
        if not sync_blocking and not ptr.is_cuda:
            seq_lens = tuple((ptr[1:] - ptr[:-1]).tolist())   # pure-CPU, no GPU sync

        # non_blocking=True overlaps the H2D copy with compute when the source is
        # pinned (pin_memory engages automatically for num_workers>0); it's a safe
        # no-op for unpinned/num_workers=0. CUDA stream ordering guarantees the
        # consuming kernels wait for the copy, so this stays correct.
        particles    = particles.to(self.device, non_blocking=True)
        y            = y.to(self.device, non_blocking=True)
        tokens       = tokens.to(self.device, non_blocking=True)
        order_labels = order_labels.to(self.device, non_blocking=True)
        ptr          = ptr.to(self.device, non_blocking=True)
        process_ids  = process_ids.to(self.device, non_blocking=True)

        y_pred = self.model(
            particles, tokens,
            mean         = self.mom_mean[0],
            std          = self.mom_std[0],
            ptr          = ptr,
            order_labels = order_labels,
            seq_lens     = seq_lens,
            process_ids  = process_ids,
        )  # (B, out_channels)

        loss_agg = self.cfg.training.get("loss_aggregation", "mean")
        loss = self._aggregate_per_process_loss(y_pred, y, process_ids, loss_agg)

        reg         = self.regularization_lambda * self.regularization(self.model)
        # Keep the no-reg loss as a detached tensor instead of syncing here with
        # .item() *before backward* — that sync forces the whole forward to finish
        # and the CPU to wait before backward is even queued. Callers materialise it
        # later (post-step, fused into one sync). See base_experiment._step.
        loss_no_reg = loss.detach()
        loss        = loss + reg
        if sync_blocking:
            assert torch.isfinite(loss).all()   # original per-step guard (D2H sync)
        return loss, loss_no_reg, None

    def _per_event_loss(self, y_pred, y, sigma=None):
        """Per-event loss vector (B,) with NO cross-event reduction.

        Mirrors the elementwise term of each loss in `_init_loss`, then means over
        the feature dim only — so the result can be segment-averaged per process.
        Because every supported loss is a mean over an elementwise term, this works
        for any of them (MSE / L1 / LogCosh / RelL1 / HETEROSC); wiring het loss into
        the LLoCa path later needs no change here (just split `sigma` out of `y_pred`
        upstream and pass it in).

        Arg order matches the existing `self.loss(y_pred, y)` call in the loop, so
        results stay numerically identical to the previous per-process loop —
        including RelL1's quirk of using the *prediction* in the denominator.
        """
        name = self.cfg.training.loss
        if name == "MSE":
            elem = (y_pred - y) ** 2
        elif name == "L1":
            elem = (y_pred - y).abs()
        elif name == "LogCosh":
            d = y_pred - y
            elem = d + torch.nn.functional.softplus(-2.0 * d) - math.log(2.0)
        elif name == "RelL1":
            eps = torch.tensor(1e-8, device=y.device, dtype=y.dtype)
            # denominator uses y_pred to match self.loss(y_pred, y) call order
            elem = ((y_pred - y) / torch.maximum(y_pred.abs(), eps)).abs()
        elif name == "HETEROSC":
            assert sigma is not None, "HETEROSC per-event loss requires sigma"
            sigma_c = torch.clamp(sigma, min=1e-15, max=1e5)
            elem = ((y - y_pred) ** 2) / (2 * sigma_c ** 2) + torch.log(sigma_c)
        else:
            raise ValueError(f"Unknown loss function {name}")
        # mean over feature dims → (B,)
        return elem.flatten(1).mean(dim=1) if elem.dim() > 1 else elem

    def _aggregate_per_process_loss(self, y_pred, y, process_ids, loss_agg, sigma=None):
        """Mean (or geometric mean) over per-process mean losses.

        Vectorised replacement for the old
            for p in torch.unique(process_ids): self.loss(y_pred[mask], y[mask])
        loop, which launched ~3 kernels per process every step and forced a
        torch.unique sync. Here we compute a per-event loss once, then segment-mean
        it by process with a single index_add_ into a fixed-size buffer
        (size = self.n_datasets, since process_ids are contiguous 0..n_datasets-1),
        so there is no Python loop and no GPU→CPU sync.

        Set LLOCA_PROC_LOSS=loop to fall back to the original loop (A/B checks).
        """
        if os.environ.get("LLOCA_PROC_LOSS", "vectorized") == "loop":
            unique_procs = torch.unique(process_ids)
            per_proc = [self.loss(y_pred[process_ids == p], y[process_ids == p])
                        for p in unique_procs]
            if loss_agg == "geometric_mean" and len(per_proc) > 1:
                return torch.stack(per_proc).log().mean().exp()
            return torch.stack(per_proc).mean()

        per_event = self._per_event_loss(y_pred, y, sigma=sigma)        # (B,)
        n_proc = self.n_datasets
        sums   = per_event.new_zeros(n_proc)
        counts = per_event.new_zeros(n_proc)
        sums.index_add_(0, process_ids, per_event)
        counts.index_add_(0, process_ids, torch.ones_like(per_event))
        present   = counts > 0
        n_present = present.sum().clamp(min=1)
        proc_mean = sums / counts.clamp(min=1)                         # 0 where a process is absent
        if loss_agg == "geometric_mean":
            # mean over present processes of log(proc_mean), then exp; absent → 0 (dropped)
            log_pm = torch.where(present, proc_mean.clamp(min=1e-30).log(),
                                 torch.zeros_like(proc_mean))
            return (log_pm.sum() / n_present).exp()
        return proc_mean.sum() / n_present


    def _init_metrics(self):
        result_key = (
            "combined" if len(self.cfg.data.dataset) > 1 else self.cfg.data.dataset[0]
        )
        return {f"{result_key}.mse": []}

    @staticmethod
    def _robust_slope(log_c, log_l):
        """Theil–Sen slope: median of pairwise slopes.  Far less sensitive than
        least-squares to the noise in short (compute, loss) windows — a single
        outlier validation can't swing the estimated exponent.  Returns None when
        no pair has a usable compute separation.
        """
        m = len(log_c)
        slopes = []
        for i in range(m):
            for j in range(i + 1, m):
                dc = log_c[j] - log_c[i]
                if abs(dc) > 1e-9:
                    slopes.append((log_l[j] - log_l[i]) / dc)
        if not slopes:
            return None
        return float(np.median(slopes))

    def _alpha_and_se(self, ds_name: str, proc_idx: int, start_idx=None,
                      target_span: float = 0.4, max_lookback: int = 30,
                      min_obs: int = 4, min_span: float = 0.1):
        """Local exponent α and its standard error over a (compute, EMA-loss) window.

        α = −slope of log(L) vs log(compute) via Theil–Sen (robust).  SE(α) is the
        OLS-style slope standard error using a robust residual scale (MAD).  A narrow
        or noisy window → large SE, so the significance tests in the controller
        default to "keep feeding" whenever the data can't support a confident call.

        Window selection is **span-targeted, not count-based**: compute accrues
        roughly linearly, so a fixed number of recent validations shrinks in
        log-compute over training and eventually falls below `min_span`, blinding the
        controller (everything looks undecidable → uniform).  Instead the recent
        window walks back until it spans `target_span` in log-compute (≥ min_obs
        points, ≤ max_lookback validations).  When start_idx is given (a probe), the
        window is exactly [start_idx:], growing as the probe continues.

        Returns (alpha, se, n_points); se = inf when the window can't be judged
        (< min_obs points, or log-compute span below `min_span`).  The span guard is
        essential — on a 2–3 point window the MAD residual scale collapses to ≈0 and
        SE looks deceptively tiny, which would let status flip on pure noise.
        """
        snapshots   = self.train_sampler.compute_snapshots
        losses_hist = getattr(self, "_proc_ema_losses", {}).get(ds_name, [])
        end = min(len(losses_hist), len(snapshots))

        def _pt(k):
            c = snapshots[k].get(proc_idx, 0)
            l = losses_hist[k]
            return (np.log(c), np.log(l)) if (c > 0 and l > 0) else None

        pts = []
        if start_idx is not None:                    # probe window: exactly [start:end)
            for k in range(max(0, start_idx), end):
                p = _pt(k)
                if p:
                    pts.append(p)
        else:                                        # recent: span-targeted lookback
            for k in range(end - 1, max(-1, end - max_lookback) - 1, -1):
                p = _pt(k)
                if p:
                    pts.append(p)
                    if len(pts) >= min_obs and (pts[0][0] - pts[-1][0]) >= target_span:
                        break
        if len(pts) < min_obs:
            return 0.0, np.inf, len(pts)
        log_c = np.array([p[0] for p in pts])
        log_l = np.array([p[1] for p in pts])
        sd_c = float(log_c.std())
        if sd_c < 1e-6 or float(log_c.max() - log_c.min()) < min_span:
            return 0.0, np.inf, len(pts)            # too little compute spread to judge
        slope = self._robust_slope(log_c, log_l)
        if slope is None:
            return 0.0, np.inf, len(pts)
        intercept = float(np.median(log_l - slope * log_c))
        resid = log_l - (intercept + slope * log_c)
        sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if sigma <= 0:
            sigma = float(resid.std())            # MAD degenerate → plain std
        se = sigma / (sd_c * np.sqrt(len(pts)))
        return float(-slope), float(se), len(pts)

    def _compute_sampler_weights(self, dataset_names):
        """Significance-based plateau detection + status-aware weighting.

        For each dataset the controller fits the local exponent α and its standard
        error SE(α) over recent (compute, EMA-loss) history.  A dataset is only
        *confirmed plateaued* — and thus allowed to fall below uniform — when, even
        optimistically (α + k·SE), its improvement rate is below a meaningful floor
        α_min; it is sent to a *probe* (boosted sampling, which shrinks SE) before
        that call is trusted, and re-probed periodically in case it is a late
        bloomer.  When the data is too noisy/narrow to decide (large SE), the test
        fails safe to "keep feeding", which is what breaks the self-reinforcing
        starvation of a transient low slope.

        The single significance knob k (sampler_sig_k) and the meaning-of-flat knob
        α_min (sampler_alpha_min) replace the old hand-thresholds (plateau_rel,
        plateau_alpha, dwell, alpha_smooth, probe_len, probe_min_growth,
        recover_min_drop): the noise handling is now adaptive per dataset.

        Returns (weights, statuses, alphas) where alphas are the clipped α used for
        weighting (handy for logging).
        """
        cfg        = self.cfg.training
        warmup     = cfg.get("sampler_warmup_vals", 10)
        reprobe    = cfg.get("sampler_reprobe_every", 20)
        probe_boost = cfg.get("sampler_probe_boost", 1.0)
        min_frac   = cfg.get("sampler_min_alpha_frac", 0.25)
        alpha_min  = cfg.get("sampler_alpha_min", 0.05)
        sig_k      = cfg.get("sampler_sig_k", 2.0)
        probe_max  = cfg.get("sampler_probe_max", 15)
        max_lookback = cfg.get("sampler_alpha_window", 30)
        gamma      = cfg.get("sampler_deficit_gamma", 1.5)
        dcap       = cfg.get("sampler_deficit_cap", 2.0)
        plat_floor = cfg.get("sampler_plateau_floor", 0.3)

        # debounce: consecutive low-α readings required before opening a probe, so a
        # single borderline/knee reading can't start (or re-start) one.  Structural,
        # not a sensitivity knob.
        TRIGGER_DEBOUNCE = 2

        if not hasattr(self, "_ds_status"):
            self._ds_status      = {n: "scaling" for n in dataset_names}
            self._probe_start    = {n: None for n in dataset_names}
            self._last_probe_val = {n: 0 for n in dataset_names}
            self._low_streak     = {n: 0 for n in dataset_names}
            self._ds_status_hist = {n: [] for n in dataset_names}
            # per-dataset *expected solo* exponent α_solo (real, from solo runs).
            # A dataset whose live local α sits below α_solo is under-scaling →
            # boosted.  Falls back to {} (no boost) if the reference is absent.
            self._solo_alpha = {}
            p = cfg.get("sampler_solo_alpha_path", None)
            if not p:
                dcfg = self.cfg.get("data", None)
                dp = (dcfg.get("data_path", "") if dcfg else "") or ""
                p = os.path.join(dp, "solo_alpha.json") if dp else None
            try:
                if p and os.path.exists(p):
                    self._solo_alpha = json.load(open(p))
                    LOGGER.info(f"Sampler loaded solo-α reference ({len(self._solo_alpha)} ds): {p}")
                else:
                    LOGGER.warning(f"Sampler solo-α reference not found at {p}; deficit boosting disabled")
            except Exception as e:
                LOGGER.warning(f"Sampler solo-α load failed ({p}): {e}")

        # current 0-based validation index (histories appended once per validation)
        hist_lens = [len(self._proc_ema_losses[m])
                     for m in dataset_names if m in self._proc_ema_losses]
        n = (max(hist_lens) - 1) if hist_lens else 0

        # recent-window (α, SE) per dataset — drives both trigger and weighting
        ase = {name: self._alpha_and_se(name, i, max_lookback=max_lookback)
               for i, name in enumerate(dataset_names)}
        # clipped α (positive, bounded) used only for the proportional weighting
        alpha_w = {name: float(np.clip(ase[name][0], 0.05, 2.0)) for name in dataset_names}

        # warmup: uniform sampling so every dataset can enter its scaling regime
        # before any down-weighting decision is made.
        if n < warmup:
            for name in dataset_names:
                self._ds_status_hist[name].append(self._ds_status[name])
            return [1.0] * len(dataset_names), dict(self._ds_status), alpha_w

        # ── status state machine (significance-gated) ─────────────────────────
        for i, name in enumerate(dataset_names):
            a, se, _ = ase[name]
            s = self._ds_status[name]
            if s == "scaling":
                # trigger a probe only after the point estimate has sat below the
                # meaningful floor for TRIGGER_DEBOUNCE consecutive validations
                if np.isfinite(se) and a < alpha_min:
                    self._low_streak[name] += 1
                else:
                    self._low_streak[name] = 0
                if self._low_streak[name] >= TRIGGER_DEBOUNCE:
                    s = "probing"
                    self._probe_start[name] = n
                    self._low_streak[name] = 0
            elif s == "probing":
                # judge on the probe window (boosted sampling has shrunk its SE)
                pa, pse, _ = self._alpha_and_se(name, i, start_idx=self._probe_start[name])
                elapsed = n - self._probe_start[name]
                # plateau requires the loss to be confidently FLAT — below the floor
                # AND not rising (pa > -alpha_min).  A rising loss (pa << 0) is a
                # disruption/spike, not a plateau, and must keep being fed.
                if np.isfinite(pse) and pa - sig_k * pse > alpha_min:
                    s = "scaling"                 # significantly still improving
                elif np.isfinite(pse) and pa + sig_k * pse < alpha_min and pa > -alpha_min:
                    s = "plateaued"               # significantly flat, not rising
                    self._last_probe_val[name] = n
                elif elapsed >= probe_max:        # undecided too long → fail safe
                    if np.isfinite(pse) and pa < alpha_min and pa > -alpha_min:
                        s = "plateaued"
                        self._last_probe_val[name] = n
                    else:
                        s = "scaling"
            elif s == "plateaued":
                if n - self._last_probe_val[name] >= reprobe:
                    s = "probing"                 # periodic re-probe for late bloomers
                    self._probe_start[name] = n
            self._ds_status[name] = s

        # ── deficit-based weights: boost datasets scaling BELOW their solo α ──
        # deficit_d = max(0, α_solo_d − α_local_d): how far a dataset is under its
        # expected solo scaling (i.e. starved by the mixture).  At-or-above-solo
        # datasets stay at baseline (they don't need extra compute); confirmed-
        # plateaued ones drop to plat_floor.  This replaces the old weight∝α rule,
        # which boosted the *fast* (easy, already-above-solo) datasets — exactly the
        # ones that need help least.
        deficits = {}
        weights = []
        for name in dataset_names:
            a_local, se, _ = ase[name]
            s = self._ds_status[name]
            sa = self._solo_alpha.get(name)
            if sa is not None and np.isfinite(se):
                deficits[name] = float(np.clip(sa - a_local, 0.0, dcap))
            else:
                deficits[name] = 0.0
            if s == "plateaued":
                w = plat_floor                       # truly done → save compute
            else:
                w = 1.0 + gamma * deficits[name]     # baseline 1.0, boosted if under solo
            weights.append(w)

        for name in dataset_names:
            self._ds_status_hist[name].append(self._ds_status[name])

        # expose deficits for logging
        self._last_deficits = deficits
        return weights, dict(self._ds_status), alpha_w

    def _validate(self, step):
        """Override base _validate.

        Single dataset: delegates to base (no change).
        Multiple datasets: runs per-process val loaders only.
        Combined loss is the geometric mean of per-process losses (scale-invariant,
        treats relative improvements equally across processes of different difficulty).
        Sampler weights are updated by estimated local power-law exponent α_eff per
        process — giving more batch slots to processes that are still improving and
        fewer to those that have plateaued.
        """
        if not self.proc_val_loaders:
            return super()._validate(step)

        # Snapshot compute counts at the start of this validation (before any updates)
        if self.train_sampler is not None:
            self.train_sampler.record_snapshot()

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
                        losses_no_reg.append(loss_no_reg.item())   # now a detached tensor
                    if mse_val is not None:
                        mse_vals.append(mse_val)
                proc_losses[name]        = float(np.mean(losses))
                proc_losses_no_reg[name] = float(np.mean(losses_no_reg)) if losses_no_reg else None
                proc_mse_vals[name]      = float(np.mean(mse_vals))      if mse_vals      else None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss_agg = self.cfg.training.get("loss_aggregation", "mean")

        def _combine(vals):
            if loss_agg == "geometric_mean":
                return float(np.exp(np.mean(np.log(np.clip(vals, 1e-10, None)))))
            return float(np.mean(vals))

        # Combine the per-process *no-reg* MSEs first, then add the whole-model reg
        # term ONCE — mirroring the training loss  GM_p(MSE_p) + reg.  Previously the
        # combined val loss folded reg into each process before the geometric mean,
        # GM_p(MSE_p + reg), a larger (reg-inflated) quantity not comparable to
        # train_loss.  (val_loss_no_reg = GM_p(MSE_p) is unchanged, so checkpoint
        # selection / HPO, which use the no-reg loss, are unaffected.)
        lnr = [v for v in proc_losses_no_reg.values() if v is not None]
        if lnr:
            val_loss_no_reg = _combine(lnr)
            reg_val  = float(self.regularization_lambda * self.regularization(self.model))
            val_loss = val_loss_no_reg + reg_val
        else:
            val_loss_no_reg = None
            val_loss = _combine(list(proc_losses.values()))

        if (step + 1) % self.cfg.training.validate_every_n_steps == 0:
            self.val_loss.append(val_loss)
            if val_loss_no_reg is not None:
                self.val_loss_no_reg.append(val_loss_no_reg)
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
        for name, loss in proc_losses_no_reg.items():
            if loss is not None:
                if name not in self.proc_val_losses_no_reg:
                    self.proc_val_losses_no_reg[name] = []
                self.proc_val_losses_no_reg[name].append(loss)

        LOGGER.info(
            f"Val loss (combined): {val_loss:.4f} | " +
            ", ".join(f"{n}={v:.4f}" for n, v in proc_losses.items())
        )

        # Update sampler weights (_compute_sampler_weights): each dataset's live
        # local exponent α is fit from recent (compute, EMA-loss) history and
        # compared to its expected SOLO exponent α_solo.  Datasets scaling BELOW
        # solo (deficit = α_solo − α_local > 0) are under-scaling/starved and get
        # boosted ∝ deficit; at-or-above-solo datasets stay at baseline; datasets
        # confirmed flat (plateaued, via the significance probe) drop to a floor.
        # Uniform during warmup.
        if self.train_sampler is not None:
            dataset_names = list(self.cfg.data.dataset)

            # EMA-smooth the per-process validation losses before fitting α.
            # sampler_alpha_ema is the decay: high = slow to forget, low = reactive.
            ema_decay = self.cfg.training.get("sampler_alpha_ema", 0.7)
            if not hasattr(self, '_proc_loss_ema'):
                self._proc_loss_ema   = {}   # running EMA scalar per process (with reg)
                self._proc_ema_losses = {}   # history of EMA loss values for α fit
                # parallel EMA of the no-reg loss (same decay) — for plotting only;
                # the sampler fits α on the with-reg EMA above.
                self._proc_loss_ema_no_reg   = {}
                self._proc_ema_losses_no_reg = {}

            for name, loss in proc_losses.items():
                if name not in self._proc_loss_ema:
                    self._proc_loss_ema[name] = loss
                else:
                    self._proc_loss_ema[name] = (
                        ema_decay * self._proc_loss_ema[name] + (1.0 - ema_decay) * loss
                    )
                self._proc_ema_losses.setdefault(name, []).append(self._proc_loss_ema[name])

            for name, loss in proc_losses_no_reg.items():
                if loss is None:
                    continue
                if name not in self._proc_loss_ema_no_reg:
                    self._proc_loss_ema_no_reg[name] = loss
                else:
                    self._proc_loss_ema_no_reg[name] = (
                        ema_decay * self._proc_loss_ema_no_reg[name] + (1.0 - ema_decay) * loss
                    )
                self._proc_ema_losses_no_reg.setdefault(name, []).append(self._proc_loss_ema_no_reg[name])

            weights, statuses, alphas = self._compute_sampler_weights(dataset_names)
            self.train_sampler.set_weights(weights)

            log_every = self.cfg.training.get("sampler_log_every_n_vals", 10)
            n_vals = len(self.val_loss)
            if log_every > 0 and n_vals % log_every == 0:
                defs = getattr(self, "_last_deficits", {})
                LOGGER.info(
                    "Sampler [α|deficit|status]: " +
                    ", ".join(f"{short_ds_name(n)}={alphas[n]:.2f}|{defs.get(n,0):.2f}|{statuses[n][:4]}"
                              for n in dataset_names)
                )

        return val_loss

    def _result_extra(self) -> dict:
        """Return per-dataset val_losses and compute counts at the best checkpoint.

        proc_val_losses        : {ds_name: val_loss}        at the best combined checkpoint step
        proc_val_losses_no_reg : {ds_name: val_loss_no_reg} at the best combined checkpoint step
        compute_ds             : {ds_name: n_samples}       total samples seen from each dataset
        """
        if not self.proc_val_losses or not self.val_loss:
            return {}
        # Use no-reg val_loss for index selection when available (matches checkpoint selection)
        ref_series = self.val_loss_no_reg if self.val_loss_no_reg else self.val_loss
        best_idx = int(np.argmin(ref_series))

        proc = {}
        for name, losses in self.proc_val_losses.items():
            if best_idx < len(losses):
                proc[name] = float(losses[best_idx])
        if not proc:
            return {}

        result = {"proc_val_losses": proc}

        proc_nr = {}
        for name, losses in self.proc_val_losses_no_reg.items():
            if best_idx < len(losses):
                proc_nr[name] = float(losses[best_idx])
        if proc_nr:
            result["proc_val_losses_no_reg"] = proc_nr

        if self.train_sampler is not None:
            dataset_names = list(self.cfg.data.dataset)
            result["compute_ds"] = {
                name: int(self.train_sampler.samples_per_process.get(i, 0))
                for i, name in enumerate(dataset_names)
            }

        return result

