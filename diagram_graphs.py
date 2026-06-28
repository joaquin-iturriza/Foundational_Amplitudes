"""Turn ``<process>.diagrams.json`` sidecars into batched graph tensors.

Each physics process owns a *set* of Feynman diagrams (emitted by
``tools/dump_diagrams.py``). This module loads that set and materialises, per
process, a dense padded tensor bundle that the diagram encoder
(``models/diagram_encoder.py``) consumes to produce one conditioning vector
``E(process)``.

Design choices
--------------
* **Dense, not message-passing.** Individual diagrams are tiny (a handful of
  nodes), so we pad to ``max_nodes`` and run masked attention. This matches the
  paper's *graph transformer* (attention over nodes with edge-conditioned bias,
  arXiv:2606.23791 Eq. 19) and avoids a torch_geometric dependency on the hot
  path. The big axis is the number of diagrams ``D`` (up to a few thousand for
  high-multiplicity processes); ``max_nodes`` stays small.

* **Particles encoded by physical properties, not a learned PID embedding.** A
  leg/propagator's identity enters through the project's
  ``build_property_matrix`` row for its PDG id — the *same* physical-quantity
  encoding used for real particles — so a Z- vs gamma-propagator is told apart
  by mass/charge and a new mediator slots in zero-shot. Self-conjugate
  propagators come out of MadGraph with an orientation-dependent sign
  (e.g. ``-22`` for the photon); :func:`_pdg_index` canonicalises ``pdg``→``-pdg``.

* **Vertices carry coupling order.** Interaction vertices contribute their
  (QED, QCD, ...) powers as node features, tying the diagram channel to the same
  perturbative-order information ``amp_orders`` already feeds the model.

Node feature layout (per node)::

    [ property_vector(n_prop) | is_external, is_vertex | is_in, is_out | order_powers(n_orders) ]

External nodes fill the property block (their leg particle) and the state flags;
vertex nodes fill the order-power block. Laplacian positional encoding is stored
separately (``lap_pe``) so the encoder can project it on its own axis.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from particle_ids import build_property_matrix, GLOBAL_PDG_IDX


# Default coupling-order channels exposed as vertex features, in fixed order.
DEFAULT_ORDER_KEYS = ("QED", "QCD")


def _pdg_index(pdg, pdg_to_idx):
    """Row index of ``pdg`` in the property matrix, canonicalising sign.

    MadGraph orients self-conjugate propagators arbitrarily (``-22`` photon,
    ``-23`` Z, ``-25`` Higgs), so fall back to ``-pdg`` when the signed id is
    absent. Returns 0 (the all-zero padding row) for an unknown id, after a
    warning — an unknown propagator should be rare and must not abort loading.
    """
    if pdg in pdg_to_idx:
        return pdg_to_idx[pdg]
    if -pdg in pdg_to_idx:
        return pdg_to_idx[-pdg]
    import warnings
    warnings.warn(f"diagram_graphs: PDG {pdg} not in property table; using zero row.")
    return 0


def _undirected_adjacency(edges, n_nodes):
    """Dense symmetric 0/1 adjacency (no self-loops) from an edge list."""
    a = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for e in edges:
        u, v = e["u"], e["v"]
        if u == v:
            continue
        a[u, v] = 1.0
        a[v, u] = 1.0
    return a


def _laplacian_pe(adj, k):
    """First ``k`` non-trivial eigenvectors of the symmetric-normalised Laplacian.

    L_sym = I - D^{-1/2} A D^{-1/2}. We drop the trivial (smallest) eigenvector
    and take the next ``k`` (the standard graph-transformer LapPE). Fewer than
    ``k`` available (small graphs) → zero-padded. Sign of each eigenvector is
    arbitrary; we leave it fixed here (callers may randomise signs per step as a
    standard LapPE augmentation).
    Returns ``(n_nodes, k)``.
    """
    n = adj.shape[0]
    pe = np.zeros((n, k), dtype=np.float32)
    if n <= 1:
        return pe
    deg = adj.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - (dinv[:, None] * adj * dinv[None, :])
    # symmetric -> eigh gives ascending real eigenvalues / orthonormal vectors
    _, vecs = np.linalg.eigh(lap)
    take = min(k, n - 1)            # skip the trivial first eigenvector
    pe[:, :take] = vecs[:, 1:1 + take]
    return pe


@dataclass
class ProcessDiagrams:
    """Dense padded graph bundle for one process's full diagram set.

    Shapes (D = #diagrams across all subprocesses, N = max nodes, ...):
      node_feat : (D, N, F_node)   float   per-node features (see module docstring)
      lap_pe    : (D, N, K)        float   Laplacian positional encoding
      node_mask : (D, N)           bool    True for real (non-padding) nodes
      edge_feat : (D, N, N, F_edge) float  per-edge features (property vec + flags); 0 if no edge
      edge_mask : (D, N, N)        bool    True where an edge exists
      leg_slot  : (D, N)           long    external leg number (1-based) per external node; -1 else
      leg_pdg   : (D, N)           long    external leg PDG per external node; 0 else
    The leg_slot/leg_pdg arrays carry the diagram↔event correspondence needed for
    the Tier-B per-event propagator-virtuality features.
    """
    node_feat: torch.Tensor
    lap_pe: torch.Tensor
    node_mask: torch.Tensor
    edge_feat: torch.Tensor
    edge_mask: torch.Tensor
    leg_slot: torch.Tensor
    leg_pdg: torch.Tensor

    @property
    def n_diagrams(self):
        return self.node_feat.shape[0]

    @property
    def f_node(self):
        return self.node_feat.shape[-1]

    @property
    def f_edge(self):
        return self.edge_feat.shape[-1]

    def to(self, device):
        return ProcessDiagrams(
            node_feat=self.node_feat.to(device),
            lap_pe=self.lap_pe.to(device),
            node_mask=self.node_mask.to(device),
            edge_feat=self.edge_feat.to(device),
            edge_mask=self.edge_mask.to(device),
            leg_slot=self.leg_slot.to(device),
            leg_pdg=self.leg_pdg.to(device),
        )


def feature_dims(n_prop, order_keys=DEFAULT_ORDER_KEYS):
    """(F_node, F_edge) for a given property width and order-key set."""
    n_orders = len(order_keys)
    f_node = n_prop + 2 + 2 + n_orders   # prop | type(2) | state(2) | orders
    f_edge = n_prop + 1                  # prop | is_external
    return f_node, f_edge


def build_process_diagrams(source, prop_matrix, k_pe=8, order_keys=DEFAULT_ORDER_KEYS,
                           max_diagrams=None):
    """Build a :class:`ProcessDiagrams` from a sidecar path/dict.

    Parameters
    ----------
    source : str | dict
        Path to ``<process>.diagrams.json`` or an already-loaded payload dict.
    prop_matrix : np.ndarray (GLOBAL_N_ENTRIES, n_prop)
        Physical-property rows indexed by ``GLOBAL_PDG_IDX`` — pass the SAME
        matrix (same smart-encoding flags) the main model uses for particles, so
        the diagram channel shares the particle encoding. Row 0 is padding.
    k_pe : int
        Laplacian PE dimension.
    order_keys : tuple[str]
        Coupling-order channels to expose as vertex features, in fixed order.
    max_diagrams : int | None
        If set, keep at most this many diagrams (a guard for high-multiplicity
        processes such as ``qqbar_Zgggg`` with >2000 diagrams). Diagrams are kept
        in MadGraph order; the drop count is reported by the caller via
        :func:`load_diagram_registry`. ``None`` keeps all.
    """
    if isinstance(source, str):
        with open(source) as f:
            payload = json.load(f)
    else:
        payload = source

    prop_matrix = np.asarray(prop_matrix, dtype=np.float32)
    n_prop = prop_matrix.shape[1]
    n_orders = len(order_keys)
    f_node, f_edge = feature_dims(n_prop, order_keys)

    # Flatten every subprocess's diagrams into one list (a process embeds over its
    # whole diagram set; multi-flavour processes like ee_qqbar pool over flavours).
    diagrams = []
    for sub in payload["subprocesses"]:
        diagrams.extend(sub["diagrams"])
    if max_diagrams is not None and len(diagrams) > max_diagrams:
        diagrams = diagrams[:max_diagrams]

    if not diagrams:
        raise ValueError(f"no diagrams in {source!r}")

    max_nodes = max(len(g["nodes"]) for g in diagrams)
    D = len(diagrams)

    node_feat = np.zeros((D, max_nodes, f_node), dtype=np.float32)
    lap_pe = np.zeros((D, max_nodes, k_pe), dtype=np.float32)
    node_mask = np.zeros((D, max_nodes), dtype=bool)
    edge_feat = np.zeros((D, max_nodes, max_nodes, f_edge), dtype=np.float32)
    edge_mask = np.zeros((D, max_nodes, max_nodes), dtype=bool)
    leg_slot = np.full((D, max_nodes), -1, dtype=np.int64)
    leg_pdg = np.zeros((D, max_nodes), dtype=np.int64)

    # node feature column offsets
    c_type = n_prop            # is_external, is_vertex
    c_state = c_type + 2       # is_in, is_out
    c_order = c_state + 2      # order powers

    for di, g in enumerate(diagrams):
        nodes, edges = g["nodes"], g["edges"]
        nn = len(nodes)
        node_mask[di, :nn] = True

        for ni, node in enumerate(nodes):
            if node["kind"] == "external":
                idx = _pdg_index(int(node["pdg"]), GLOBAL_PDG_IDX)
                node_feat[di, ni, :n_prop] = prop_matrix[idx]
                node_feat[di, ni, c_type + 0] = 1.0       # is_external
                if node.get("state") == "out":
                    node_feat[di, ni, c_state + 1] = 1.0  # is_out
                else:
                    node_feat[di, ni, c_state + 0] = 1.0  # is_in
                if node.get("leg_number") is not None:
                    leg_slot[di, ni] = int(node["leg_number"])
                    leg_pdg[di, ni] = int(node["pdg"])
            else:  # vertex
                node_feat[di, ni, c_type + 1] = 1.0       # is_vertex
                orders = node.get("orders", {})
                for oi, key in enumerate(order_keys):
                    node_feat[di, ni, c_order + oi] = float(orders.get(key, 0))

        for e in edges:
            u, v = e["u"], e["v"]
            idx = _pdg_index(int(e["pdg"]), GLOBAL_PDG_IDX)
            feat = np.zeros(f_edge, dtype=np.float32)
            feat[:n_prop] = prop_matrix[idx]
            feat[n_prop] = 1.0 if e.get("external") else 0.0
            for (a, b) in ((u, v), (v, u)):     # symmetric
                edge_feat[di, a, b] = feat
                edge_mask[di, a, b] = True

        adj = _undirected_adjacency(edges, nn)
        lap_pe[di, :nn] = _laplacian_pe(adj, k_pe)

    return ProcessDiagrams(
        node_feat=torch.from_numpy(node_feat),
        lap_pe=torch.from_numpy(lap_pe),
        node_mask=torch.from_numpy(node_mask),
        edge_feat=torch.from_numpy(edge_feat),
        edge_mask=torch.from_numpy(edge_mask),
        leg_slot=torch.from_numpy(leg_slot),
        leg_pdg=torch.from_numpy(leg_pdg),
    )


def load_diagram_registry(diagrams_dir, process_names, spin_onehot=False,
                          color_onehot=False, is_massless=False, standardize=False,
                          k_pe=8, order_keys=DEFAULT_ORDER_KEYS, max_diagrams=None,
                          logger=None):
    """Load a ``{process_name: ProcessDiagrams}`` registry for the given processes.

    The property matrix is built ONCE with the supplied smart-encoding flags (pass
    the same ones the experiment uses for particles) and shared across processes.
    Missing sidecars are reported and skipped (the caller decides whether to error
    or run without a diagram for that process).
    """
    prop_matrix, _ = build_property_matrix(
        spin_onehot=spin_onehot, color_onehot=color_onehot,
        is_massless=is_massless, standardize=standardize,
    )
    registry = {}
    for name in process_names:
        path = os.path.join(diagrams_dir, f"{name}.diagrams.json")
        if not os.path.exists(path):
            msg = f"diagram_graphs: no sidecar for process '{name}' at {path}"
            if logger is not None:
                logger.warning(msg)
            else:
                import warnings
                warnings.warn(msg)
            continue
        pd = build_process_diagrams(
            path, prop_matrix, k_pe=k_pe, order_keys=order_keys,
            max_diagrams=max_diagrams,
        )
        registry[name] = pd
        if logger is not None:
            logger.info(f"  diagrams[{name}]: D={pd.n_diagrams} maxN={pd.node_feat.shape[1]} "
                        f"F_node={pd.f_node} F_edge={pd.f_edge}")
    return registry, prop_matrix.shape[1]
