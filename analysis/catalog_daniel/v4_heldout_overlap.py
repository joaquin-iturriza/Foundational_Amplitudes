"""Which of Daniel's 139 held-out processes (recipes/catalog_daniel_heldout.yaml: 96 interpolation relabels, 43
extrapolation) catalog_v4's pretraining trained on, the process itself or any relabel of it (plan dan_zeroshot_v4, D488:
"the held outs should truly be held outs"; interpolation is relabels by construction and is scored whatever this finds).

Catalog_v4's training: every entry of recipes/catalog_v4_default25_train.yaml (trial 1262's data.processes_file, read
from its config.yaml on CC; alpha_s and heavy-quark variants by their base entry, one-loop entries included), widened
to every member of a catalog_v4 equality class (analysis/catalog_v4/census_classes.json) holding a training entry, so
the |M|^2 equalities the equality step found (ee_X = uubar_X, massless flavour relabels) count as relabels.

A relabel, deliberately broad so that a process kept as held out is held out under any reading: charge conjugation
of every leg; any permutation of the down-type quarks {d, s, b} (b massless in catalog_v4); of the up-type {u, c}; of
the lepton generations (e, mu, tau with their neutrinos), each independently; initial and final legs are never swapped.
Coupling orders, tree against one loop, the b and tau masses and the sampling are ignored: a match on particle content
up to these relabels marks the process as seen.

Output: analysis/catalog_daniel/data/v4_heldout_overlap.json {process: {group, identical: [...], relabel: [...]}},
the counts printed.
    python3 analysis/catalog_daniel/v4_heldout_overlap.py
"""
import itertools, json, os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MG = {"d": 1, "u": 2, "s": 3, "c": 4, "b": 5, "t": 6, "e-": 11, "ve": 12, "mu-": 13, "vm": 14, "ta-": 15, "vt": 16,
      "g": 21, "a": 22, "z": 23, "w+": 24, "h": 25}
SELF_CONJ = {21, 22, 23, 25}


def mg_pdg(tok):
    if tok in MG:
        return MG[tok]
    anti = {"e+": "e-", "mu+": "mu-", "ta+": "ta-", "w-": "w+"}.get(tok, tok[:-1] if tok.endswith("~") else None)
    return -MG[anti]


def key(ini, fin):
    """The orbit's least (initial, final) multiset pair under the broad relabels."""
    best = None
    for dn, up, lep, cp in itertools.product(itertools.permutations((1, 3, 5)), itertools.permutations((2, 4)),
                                             itertools.permutations((11, 13, 15)), (1, -1)):
        m = {1: dn[0], 3: dn[1], 5: dn[2], 2: up[0], 4: up[1]}
        m.update({l: l2 for l, l2 in zip((11, 13, 15), lep)}); m.update({l + 1: l2 + 1 for l, l2 in zip((11, 13, 15), lep)})

        def f(p):
            q = (1 if p > 0 else -1) * m.get(abs(p), abs(p))
            return abs(q) if abs(q) in SELF_CONJ else cp * q
        k = (tuple(sorted(map(f, ini))), tuple(sorted(map(f, fin))))
        best = k if best is None or k < best else best
    return best


def main():
    procs = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_processes.yaml")))["processes"]
    procs.update(yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_loops.yaml")))["processes"])
    train = {p["name"].split("__")[0] for p in
             yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_default25_train.yaml")))["processes"]}
    classes = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "census_classes.json")))["classes"]
    seen = set(train)
    for members in classes.values():
        if train & set(members):
            seen |= set(members)
    missing = sorted(n for n in seen if n not in procs)
    assert not [n for n in train if n not in procs], "a training entry without its pdg_ids"
    by_key, by_id = {}, {}
    for n in seen - set(missing):
        ids = procs[n]["pdg_ids"]
        by_key.setdefault(key(ids[:2], ids[2:]), []).append(n)
        by_id.setdefault((tuple(sorted(ids[:2])), tuple(sorted(ids[2:]))), []).append(n)
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    draft = json.load(open(os.path.join(ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    group = {n: g for g in ("interp", "extrap") for n in draft[g]}
    held = [p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_daniel_heldout.yaml")))["processes"]]
    assert set(held) == set(group), "the recipe's 139 are the figure's 139"
    out = {}
    for n in held:
        a, b = meta[n]["process"].split(">")
        ini, fin = [mg_pdg(t) for t in a.split()], [mg_pdg(t) for t in b.split()]
        out[n] = {"group": group[n], "process": meta[n]["process"],
                  "identical": sorted(by_id.get((tuple(sorted(ini)), tuple(sorted(fin))), [])),
                  "relabel": sorted(by_key.get(key(ini, fin), []))}
    # self-check: a v4 training entry is its own relabel, and d d~ > h h relabels b b~ > h h
    t0 = sorted(train)[0]; ids = procs[t0]["pdg_ids"]
    assert t0 in by_key[key(ids[:2], ids[2:])]
    assert key([1, -1], [25, 25]) == key([5, -5], [25, 25]) == key([-3, 3], [25, 25])
    json.dump({"training_entries": len(train), "seen_with_classes": len(seen), "class_members_without_pdg": missing,
               "processes": out}, open(os.path.join(HERE, "data", "v4_heldout_overlap.json"), "w"), indent=1)
    print(f"catalog_v4 training: {len(train)} entries, {len(seen)} with their equality classes "
          f"({len(missing)} class members without pdg_ids: {missing[:5]})")
    for g in ("interp", "extrap"):
        ns = [n for n in held if group[n] == g]
        rel = [n for n in ns if out[n]["relabel"]]
        print(f"{g}: {len(ns)}; identical process in catalog_v4 training {sum(bool(out[n]['identical']) for n in ns)}; "
              f"a relabel in it {len(rel)}; truly held out {len(ns) - len(rel)}")
        if g == "extrap":
            for n in rel:
                print(f"  seen: {n} ({out[n]['process']}) <- {out[n]['relabel'][:4]}")


if __name__ == "__main__":
    main()
