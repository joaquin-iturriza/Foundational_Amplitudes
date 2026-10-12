"""Add-W children of the replay parents (D34, D37, D39): the finale fine-tuned on W+4g, W+5g with a 1% or 10% replay
buffer (tp3_waddm_buf1pc4k t17, buf10pc4k t46, buf10pc2k t68), each its best checkpoint on W and its best on all 30
(b1w, b1m, b10w, b10m), fine-tuned on Z+4g, Z+5g and ee->ttbar at D = 10^1 .. 10^3 (8k grid, tp3_waddr_<k>fte_<p>_d<k>)
and 10^4 (32k, tp3_waddr_<k>fte32k_<p>_d8). Each cell the best of its two points (expect_tp3.json): on Z the finale
cell's own points, on ee->ttbar the D25 fixed points, where the finale ran the same points (tp3_finfpfte_ee_ttbar_d*;
at 10^4 the finale cell's own). Reference per cell: the finale at the same points, scratch (Z: the first 8 trials of
tp3_hmscr[32k]; ee->ttbar: cells.scratch, the ladder's search, best_chosen at 32k), and on Z the finale+W pretraining
(tp3_fwngfte[32k]). MSE of log|M|^2 at the best checkpoint. Data: scratch_sweeps.json.
    python analysis/transfer/replay_children.py      -> figs/replay_children (gain over the finale against D); table
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S, scratch, best_chosen  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

E = json.load(open(os.path.join(ROOT, "analysis", "transfer", "expect_tp3.json")))
PROBES = [("uubar_Zgggg", r"$u\bar u\to Z+4g$"), ("uubar_Zggggg", r"$u\bar u\to Z+5g$"), ("ee_ttbar", r"$e^+e^-\to t\bar t$")]
PARENTS = [("b1w", r"replay $1\%$, best on $W$", ps.C.purple), ("b1m", r"replay $1\%$, best on all $30$", ps.C.sky),
           ("b10w", r"replay $10\%$, best on $W$", ps.C.vermillion), ("b10m", r"replay $10\%$, best on all $30$", ps.C.orange)]
KS = [2, 3, 4, 5, 6, 8]


def pts(n):
    """Best of the cell's expected points, None until both are in."""
    tr = [t for t in S.get(n, []) if t.get("val_loss") is not None and t.get("prepd_std") and t["hp"] in E.get(n, [])]
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if len(tr) == 2 else None


def finale(p, k):
    if k == 8:
        return pts(f"tp3_finfte32k_{p}_d8")
    return pts(f"tp3_finfpfte_{p}_d{k}" if p == "ee_ttbar" else f"tp3_finfte_{p}_d{k}")


def scr(p, k):
    if p == "ee_ttbar":
        return best_chosen(f"tp3_scr32k_{p}_d8") if k == 8 else scratch(p, k, steered=False)[0]
    tr = sorted((t for t in S.get(f"tp3_hmscr{'32k' if k == 8 else ''}_{p}_d{k}", []) if t.get("val_loss") is not None),
                key=lambda t: (t["order"] if t.get("order") is not None else 99, t.get("site") != "ccin2p3"))[:8]
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if len(tr) == 8 else None


def child(k_, p, k):
    return pts(f"tp3_waddr_{k_}fte32k_{p}_d8" if k == 8 else f"tp3_waddr_{k_}fte_{p}_d{k}")


def wng(p, k):
    if p == "ee_ttbar":
        return None
    return pts(f"tp3_fwngfte32k_{p}_d8" if k == 8 else f"tp3_fwngfte_{p}_d{k}")


dex = lambda a, b: None if a is None or b is None else math.log10(a / b)
fmt = lambda x: "    -" if x is None else f"{x:+.2f}"
for p, _ in PROBES:
    print(p, "  gain over the finale's child (dex); last columns: finale and finale+W over scratch")
    for k in KS:
        f, s = finale(p, k), scr(p, k)
        print(f"  10^{k / 2:<4}" + " ".join(f"{k_:>5s} {fmt(dex(f, child(k_, p, k)))}" for k_, _, _ in PARENTS)
              + f"   finale {fmt(dex(s, f))}  +W {fmt(dex(s, wng(p, k)))}")

fig = ps.panels(3)
for (fg, ax), (p, lab) in zip(fig, PROBES):
    for k_, l, c in PARENTS:
        xs = [(k / 2, dex(finale(p, k), child(k_, p, k))) for k in KS]
        xs = [(x, y) for x, y in xs if y is not None]
        ax.plot([x for x, _ in xs], [y for _, y in xs], "o-", color=c, label=l)
    xs = [(k / 2, dex(finale(p, k), wng(p, k))) for k in KS]
    xs = [(x, y) for x, y in xs if y is not None]
    if xs:
        ax.plot([x for x, _ in xs], [y for _, y in xs], "s--", color=ps.C.green, label=r"finale$+W$ (pretrained)")
    ax.axhline(0, color=ps.C.blue, label="finale")
    ax.set_xlabel(r"$\log_{10} D$")
    ax.set_ylabel(r"$\log_{10}(L_\mathrm{finale}/L)$")
    ps.process_label(ax, lab)
ps.legend(fig[0][1], "upper left")
ps.save_panels(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "replay_children"))
