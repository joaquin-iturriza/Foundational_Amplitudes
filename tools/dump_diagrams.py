#!/usr/bin/env python
"""Dump the Feynman-diagram graph of each physics process to a JSON sidecar.

For every process in the dataset registry (``mg5_pipeline_final.PROCESSES``) this
drives MadGraph5 symbolically — ``import model`` + the process's ``generate`` /
``add process`` commands — and serialises, per diagram, a small graph:

    nodes : external-particle endpoints + interaction vertices
    edges : external legs + internal propagators, each carrying a PDG id

This is exactly the diagram-as-graph representation of arXiv:2606.23791 (their
Eq. 10), but here the edge/leg PDG ids are kept raw so the *graph loader* can map
them through the project's physical-property table (``particle_ids``) — the same
encoding used for real particles — instead of a learned PDG embedding.

It is fully static (one symbolic ``generate`` per process, seconds, no event
generation) and CPU-only, so it runs on a login node. The output sidecar is
``<out_dir>/<process>.diagrams.json``; downstream code (``diagram_graphs.py``)
turns it into batched graph tensors.

Usage:
    python tools/dump_diagrams.py --process ee_ttbar --process ee_uug
    python tools/dump_diagrams.py --all
    python tools/dump_diagrams.py --all --out-dir data/diagrams
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys

# MadGraph lives next to the conda env on Jean Zay; honour the same env vars the
# data pipeline uses (MG5_BIN points at <root>/bin/mg5_aMC, so the package root is
# its grandparent), falling back to $WORK/mg5amcnlo.
_WORK = os.environ.get("WORK", "")
_MG5_ROOT = os.environ.get("MG5_WORK_DIR") or (f"{_WORK}/mg5amcnlo" if _WORK else None)
if _MG5_ROOT and _MG5_ROOT not in sys.path:
    sys.path.insert(0, _MG5_ROOT)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Default sidecar location: data/diagrams/ in the project tree (versionable, small).
DEFAULT_OUT_DIR = os.path.join(_HERE, "data", "diagrams")


def _silence_mg5():
    """MadGraph chatters on stdout and via logging; quiet both for batch use."""
    logging.disable(logging.CRITICAL)


@contextlib.contextmanager
def _quiet():
    """Swallow MG5's banner / per-command stdout (it cannot be turned off via API)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _master_cmd():
    import madgraph.interface.master_interface as mi
    return mi.MasterCmd()


def _orders_for_vertex(model, interaction_id):
    """Coupling-order dict for an interaction vertex, e.g. {'QED': 1} or {'QCD': 1}.

    The external-leg endpoints carry MadGraph interaction id 0 (no interaction);
    return an empty dict for those.
    """
    if interaction_id == 0:
        return {}
    inter = model.get("interaction_dict").get(interaction_id)
    if inter is None:
        return {}
    return {str(k): int(v) for k, v in dict(inter.get("orders")).items()}


def _diagram_to_graph(diagram, model):
    """Convert one MadGraph diagram into a ``{nodes, edges}`` dict.

    Uses ``madgraph.core.drawing.FeynmanDiagram`` to resolve the diagram into a
    concrete set of vertices (``vertexList``) and lines (``lineList``) with proper
    begin/end connectivity — the same machinery MG5 uses to draw diagrams — rather
    than hand-reconstructing propagators from the raw vertex/leg encoding.

    Node kinds:
      - "external" : a true external leg endpoint (MG5 vertex id 0). Carries the
                     external leg ``number`` (1,2 = initial; 3.. = final), its
                     ``pdg`` and ``state`` — this ``number`` is the slot index used
                     to map a leg onto the event's particle for per-event features.
      - "vertex"   : an interaction vertex. Carries its ``interaction_id`` and the
                     coupling ``orders`` (QED/QCD powers).

    Edges mirror ``lineList``: every line is an edge ``(u, v)`` between node
    indices, carrying the line ``pdg``, an ``external`` flag, and the leg
    ``number`` (meaningful for external legs; kept for propagators too).
    """
    import madgraph.core.drawing as drawing

    fd = drawing.FeynmanDiagram(diagram, model)
    fd.main()  # populates vertexList / lineList and their begin/end links

    vidx = {id(v): i for i, v in enumerate(fd.vertexList)}

    # Map each external endpoint vertex -> the external line touching it, so the
    # endpoint node can carry that leg's pdg/number/state.
    ext_line_for_vertex = {}
    for line in fd.lineList:
        if not line.is_external():
            continue
        for endpoint in (line.begin, line.end):
            if endpoint is not None and endpoint.id == 0:
                ext_line_for_vertex[id(endpoint)] = line

    nodes = []
    for v in fd.vertexList:
        if v.id == 0:
            line = ext_line_for_vertex.get(id(v))
            nodes.append({
                "kind": "external",
                "interaction_id": 0,
                "leg_number": int(line.number) if line is not None else None,
                "pdg": int(line.id) if line is not None else None,
                "state": "out" if (line is not None and line.state) else "in",
            })
        else:
            nodes.append({
                "kind": "vertex",
                "interaction_id": int(v.id),
                "orders": _orders_for_vertex(model, v.id),
            })

    edges = []
    for line in fd.lineList:
        u = vidx.get(id(line.begin))
        w = vidx.get(id(line.end))
        if u is None or w is None:
            # Defensive: a dangling line should not happen once main() has run.
            continue
        edges.append({
            "u": u,
            "v": w,
            "pdg": int(line.id),
            "external": bool(line.is_external()),
            "leg_number": int(line.number),
        })

    return {"nodes": nodes, "edges": edges}


def _external_legs(amplitude):
    """Ordered external-leg list ``[{number, pdg, state}]`` for a subprocess.

    ``state`` is False for initial-state legs in MadGraph; we expose 'in'/'out'.
    Legs are returned sorted by ``number`` (1,2 initial; 3.. final) — the canonical
    MadGraph ordering.
    """
    legs = amplitude.get("process").get("legs")
    out = []
    for leg in legs:
        out.append({
            "number": int(leg.get("number")),
            "pdg": int(leg.get("id")),
            "state": "out" if leg.get("state") else "in",
        })
    out.sort(key=lambda d: d["number"])
    return out


def dump_process(name, spec, out_dir, model_name="sm"):
    """Generate one process in MG5 and write ``<out_dir>/<name>.diagrams.json``.

    A process may expand into several subprocesses (e.g. ``ee_qqbar`` adds five
    flavours); each contributes its own external-leg list and diagram set.
    """
    mg5_generate = spec.get("mg5_generate")
    if not mg5_generate:
        return None, "no mg5_generate commands"
    model_name = spec.get("model", model_name)

    cmd = _master_cmd()
    with _quiet():
        cmd.exec_cmd(f"import model {model_name}", printcmd=False, precmd=True)
        for line in mg5_generate:
            cmd.exec_cmd(line, printcmd=False, precmd=True)

    model = cmd._curr_model
    amps = list(cmd._curr_amps)
    if not amps:
        return None, "MG5 produced no amplitudes"

    subprocesses = []
    for amp in amps:
        diagrams = amp.get("diagrams")
        graphs = [_diagram_to_graph(d, model) for d in diagrams]
        subprocesses.append({
            "process_str": amp.get("process").nice_string().replace("Process: ", ""),
            "external": _external_legs(amp),
            "n_diagrams": len(graphs),
            "diagrams": graphs,
        })

    payload = {
        "process": name,
        "model": model_name,
        "mg5_generate": list(mg5_generate),
        "n_subprocesses": len(subprocesses),
        "subprocesses": subprocesses,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.diagrams.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    n_diag = sum(s["n_diagrams"] for s in subprocesses)
    return path, f"{len(subprocesses)} subprocess(es), {n_diag} diagram(s)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--process", action="append", default=[],
                        help="Process name from the registry (repeatable).")
    parser.add_argument("--all", action="store_true",
                        help="Dump every process in the registry.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--model", default="sm",
                        help="Default MadGraph model (overridden per-process by spec['model']).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip processes whose sidecar already exists.")
    args = parser.parse_args()

    _silence_mg5()
    from mg5_pipeline_final import PROCESSES

    if args.all:
        names = list(PROCESSES.keys())
    else:
        names = args.process
    if not names:
        parser.error("specify --process NAME (repeatable) or --all")

    ok, failed = [], []
    for name in names:
        if name not in PROCESSES:
            print(f"[skip] {name}: not in registry")
            failed.append(name)
            continue
        out_path = os.path.join(args.out_dir, f"{name}.diagrams.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] {name}: sidecar exists")
            ok.append(name)
            continue
        try:
            path, info = dump_process(name, PROCESSES[name], args.out_dir, args.model)
        except Exception as e:  # one bad process shouldn't abort the batch
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed.append(name)
            continue
        if path is None:
            print(f"[FAIL] {name}: {info}")
            failed.append(name)
        else:
            print(f"[ok]   {name}: {info} -> {os.path.relpath(path, _HERE)}")
            ok.append(name)

    print(f"\nDone: {len(ok)} ok, {len(failed)} failed"
          + (f" ({', '.join(failed)})" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
