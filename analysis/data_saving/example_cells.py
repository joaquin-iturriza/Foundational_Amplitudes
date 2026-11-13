"""The trials behind the cells of old_setup_examples (the draft's data-saving figure, D433) that its representative
examples re-run with per-event predictions (D444): per probe and D, the scratch cell's best trial and the
every-structure fine-tune's (tp3_finfte) trial, picked exactly as cells.scratch and finale_synth.cell pick them.
    python analysis/data_saving/example_cells.py   -> data/example_cells.json (sweep, hp, site, loss per cell)
"""
import io, contextlib, json, os, sys
os.environ["TP_USE_32K"] = "0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "transfer"))
with contextlib.redirect_stdout(io.StringIO()):
    from cells import S, ARM, scratch  # noqa: E402
    from finale_synth import cell  # noqa: E402

PROBES = ["uubar_Zggg", "ee_ddbarg", "udbar_enu", "ee_bb_nlo"]
KS = [4, 6, 8]                      # D = 10^2, 10^3, 10^4


def tagged(names):
    return [(n.removesuffix("_002") if n.endswith("_002") else n, n, t) for n in names for t in S.get(n, [])]


def scratch_trial(p, k):
    loss, _ = scratch(p, k, steered=False)
    names = [f"{ARM[p]}_{p}_d{k}", f"tp3_scrh_{p}_d{k}", f"tp3_scrh10_{p}_d{k}"]
    hits = [(n, t) for _, n, t in tagged([m for b in names for m in (b, b + "_002")])
            if t.get("val_loss") is not None and t.get("prepd_std") and abs(t["val_loss"] * t["prepd_std"] ** 2 - loss) < 1e-12 * max(loss, 1)]
    assert hits, (p, k)
    return loss, hits[0]


def ft_trial(p, k, fam="tp3_finfte"):
    loss = cell(fam, p, k)
    names = [f"{fam}_{p}_d{k}_lad", f"{fam}_{p}_d{k}", f"{fam}_{p}_d{k}_002"]
    hits = [(n, t) for _, n, t in tagged(names)
            if t.get("val_loss") is not None and t.get("prepd_std") and abs(t["val_loss"] * t["prepd_std"] ** 2 - loss) < 1e-12 * max(loss, 1)]
    assert hits, (p, k)
    return loss, hits[0]


out = []
for p in PROBES:
    for k in KS:
        for arm, f in (("scratch", scratch_trial), ("fine-tune", ft_trial)):
            loss, (n, t) = f(p, k)
            out.append(dict(probe=p, k=k, D=round(10 ** (k / 2)), arm=arm, sweep=n, hp=t["hp"], site=t.get("site"),
                            loss=loss, T=t.get("T")))
            print(out[-1])
json.dump(out, open(os.path.join(ROOT, "analysis", "data_saving", "data", "example_cells.json"), "w"), indent=1)
