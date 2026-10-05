"""Transfer study, the HP landscape of every 64k-step pretraining (the ladder's rungs 1-9, the ee_uu pretraining, the six
star arms), to judge the pretraining protocol (hp47 and hp73 from 2026-10-02, hp73 alone from 2026-10-04). Every sweep draws from the same
candidate pool, so an hp index is the same point on every pretraining. Data: analysis/transfer/pretrain_hp.json
(collect_ladder.py tp3_ladder_r tp3_star_ tp3_pre64_ on every site, merged; a trial that trained to 64k and could not
write its result, or diverged, carries its best checkpoint rescored, ladder_eval_best.json / star_eval_best.json).
Every value: val_loss_no_reg at the best checkpoint over the best of its pretraining. A diverged trial not rescored yet
has no value to draw: it is listed with its divergence step, as is every stopped trial (cancelled or lost, no result).
  <base>_rank       per pretraining, each trial over the pretraining's best; hp15, hp47, hp73, hp95 marked
  <base>_landscape  the same ratio against lr, lambda, warm-up fraction and EMA decay (open: EMA off)
    python analysis/transfer/pretrain_hp.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

P = json.load(open(os.path.join(ROOT, "analysis", "transfer", "pretrain_hp.json")))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs", "pretrain_hp")
LADDER = [f"tp3_ladder_r{i}" for i in range(1, 10)]
# the study's final setup: the first resonance and Sudakov arms (replaced by the W-pole and EW-Sudakov arms, which ran hp73
# alone and so rank nothing) are left out
OTHER = ["tp3_pre64_ee_uu"] + [f"tp3_star_{a}" for a in ("soft", "isr", "deadcone", "threshold")]
NAME = {**{f"tp3_ladder_r{i}": f"rung {i}" for i in range(1, 10)}, "tp3_pre64_ee_uu": r"$ee\to u\bar u$",
        "tp3_star_soft": "soft", "tp3_star_isr": "ISR", "tp3_star_deadcone": "dead cone",
        "tp3_star_resonance": "resonance", "tp3_star_threshold": "threshold", "tp3_star_sudakov": "Sudakov"}
MARK = {47: ("hp47", ps.C.vermillion, "o"), 73: ("hp73", ps.C.blue, "s"), 95: ("hp95", ps.C.orange, "D"),
        15: ("hp15", ps.C.green, "^")}
OTHER_STYLE = ("other searched points", ps.C.grey, ".")


def ratios(n):
    v = [t for t in P[n] if "val_loss" in t]
    b = min(t["val_loss"] for t in v)
    return [(t, t["val_loss"] / b) for t in v]


def style(hp):
    return MARK.get(hp, OTHER_STYLE)


fig, axes = ps.figure(ncols=2)
for ax, group in zip(axes, (LADDER, OTHER)):
    for y, n in enumerate(group):
        for t, r in ratios(n):
            _, col, m = style(t["hp"])
            ax.plot(r, y, m, color=col, mfc="none" if t["state"] != "done" else col, zorder=3 if t["hp"] in MARK else 2)
    ax.set_yticks(range(len(group)), [NAME[n] for n in group])
    ax.set_ylim(len(group) - 0.5, -0.5)
    ax.set_xscale("log"); ax.set_xlabel(r"$\mathcal{L}_{\rm val}/\mathcal{L}_{\rm val}^{\rm best}$")
for hp, (lab, col, m) in list(MARK.items()) + [(None, OTHER_STYLE)]:
    axes[0].plot([], [], m, color=col, label=lab)
axes[0].plot([], [], "o", color="k", mfc="none", label="rescored best checkpoint")
ps.shared_legend(fig, axes[0], ncol=3)
ps.save(fig, FIG + "_rank")

knobs = [("lr", "lr", True), ("lambda", r"$\lambda$", True), ("warmup", "warm-up fraction", False),
         ("ema_decay", "EMA decay", False)]
fig, axes = ps.figure(ncols=2, nrows=2)
for ax, (k, lab, lg) in zip(axes.flat, knobs):
    for n in LADDER + OTHER:
        for t, r in ratios(n):
            _, col, m = style(t["hp"])
            ax.plot(t[k], r, m, color=col, mfc=col if (k != "ema_decay" or t["ema"]) else "none",
                    zorder=3 if t["hp"] in MARK else 2)
    if lg:
        ax.set_xscale("log")
    ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel(r"$\mathcal{L}_{\rm val}/\mathcal{L}_{\rm val}^{\rm best}$")
for hp, (lab, col, m) in list(MARK.items()) + [(None, OTHER_STYLE)]:
    axes.flat[0].plot([], [], m, color=col, label=lab)
ps.shared_legend(fig, axes.flat[0], ncol=5)
ps.save(fig, FIG + "_landscape")

for n in LADDER + OTHER:
    r = {t["hp"]: x for t, x in ratios(n)}
    best = min(r, key=r.get)
    print(f"{NAME[n]:>12}: " + "  ".join(f"hp{h} {r[h]:.2f}" if h in r else f"hp{h} -" for h in MARK)
          + f"   best hp{best}, {len(r)} trials with a value")
for n in LADDER + OTHER:
    for t in P[n]:
        if "val_loss" not in t and t["state"] == "diverged":
            print(f"  not drawn, diverged: {NAME[n]} hp{t['hp']} at step {t.get('step')} (lr {t['lr']:.1e}, lambda {t['lambda']:.0e})")
print("  stopped (cancelled or lost, no result): " + ", ".join(
    f"{NAME[n]} hp{t['hp']}" for n in LADDER + OTHER for t in P[n] if "val_loss" not in t and t["state"] != "diverged"))
