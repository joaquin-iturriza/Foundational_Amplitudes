#!/usr/bin/env python3
"""Backfill `data.dataset` in run configs that were saved before the recipe resolved.

WHY. `full_run()` used to save config.yaml BEFORE `init_physics()`, and it is
`experiment._resolve_recipe_config` (inside init_physics) that replaces
`data.dataset`/`data.amp_orders` with the processes the recipe actually names. So every
run on the recipe path (`data.processes_file` set) persisted the inherited 8-entry default
from config/amplitudes.yaml instead of what it trained on. That key has repeatedly been
read as ground truth and has inverted claims about what was held out.

base_experiment.full_run now re-saves after init_physics, so NEW runs are correct. This
script repairs the ones already on disk, writing exactly what the run itself would have
written: the recipe's `name` list, in recipe order.

SAFETY.
  * Default is a dry run. `--apply` is required to write.
  * `--apply` first tars every file it will touch into a restore point.
  * A config is only rewritten when its recipe file still EXISTS and PARSES, and when the
    current `data.dataset` actually differs from the resolved list.
  * Rewrites are surgical: only the `data.dataset:` block is replaced, in place, with the
    surrounding YAML left byte-identical. Nothing else in the file is reformatted, so a
    diff shows only the key that was wrong.
  * A `data.dataset_backfilled: true` marker is added so the repair is auditable and
    re-runs are idempotent.

CAVEAT worth knowing: this reconstructs from the recipe AS IT IS NOW. If a recipe was
edited after a run used it, the backfill reflects the edited recipe. Recipes are versioned
in git, so `git log -- <recipe>` tells you whether that is a risk for a given run.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def recipe_names(path: str, cache: dict) -> list[str] | None:
    """Process names a recipe declares, in order. None if unusable."""
    if path in cache:
        return cache[path]
    out = None
    try:
        with open(path) as f:
            spec = yaml.safe_load(f)
        if isinstance(spec, dict):
            spec = spec.get("processes", spec)
        if isinstance(spec, list):
            names = [p["name"] for p in spec if isinstance(p, dict) and "name" in p]
            out = names or None
    except Exception:
        out = None
    cache[path] = out
    return out


def parse_block(text: str, key: str) -> tuple[int, int, list[str]] | None:
    """Locate a `  <key>:` list block under `data:`. Returns (start, end, items)
    as character offsets, or None. Only handles the block-list form OmegaConf dumps."""
    m = re.search(rf"^(?P<ind>[ \t]+){re.escape(key)}:[ \t]*$", text, re.M)
    if not m:
        return None
    indent = m.group("ind")
    pos = m.end()
    if pos < len(text) and text[pos] == "\n":
        pos += 1
    items, cur = [], pos
    item_re = re.compile(rf"^{re.escape(indent)}- (?P<v>.*)$")
    while cur < len(text):
        nl = text.find("\n", cur)
        line = text[cur : nl if nl != -1 else len(text)]
        im = item_re.match(line)
        if not im:
            break
        items.append(im.group("v").strip())
        cur = (nl + 1) if nl != -1 else len(text)
    if not items:
        return None
    return m.start(), cur, items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--root", default=str(REPO / "runs"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N configs (debug)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    t0 = time.time()
    cache: dict[str, list[str] | None] = {}
    scanned = recipe_runs = already_ok = to_fix = 0
    missing_recipe = unparsed = 0
    planned: list[tuple[Path, list[str], list[str]]] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not (fn.startswith("config") and fn.endswith(".yaml")):
                continue
            p = Path(dirpath) / fn
            scanned += 1
            if args.limit and scanned > args.limit:
                break
            try:
                text = p.read_text()
            except Exception:
                continue
            if "processes_file: " not in text:
                continue  # not a recipe run -> data.dataset is authoritative, leave alone
            rm = re.search(r"^\s*processes_file:\s*(\S+)\s*$", text, re.M)
            if not rm or rm.group(1) in ("null", "none", "~"):
                continue
            recipe_runs += 1
            if "dataset_backfilled: true" in text:
                already_ok += 1
                continue
            names = recipe_names(rm.group(1), cache)
            if names is None:
                if not Path(rm.group(1)).exists():
                    missing_recipe += 1
                else:
                    unparsed += 1
                continue
            # parse_block takes the FIRST `dataset:` block; refuse to guess if a
            # config somehow carries more than one, rather than rewrite the wrong key.
            if len(re.findall(r"^[ \t]+dataset:[ \t]*$", text, re.M)) != 1:
                unparsed += 1
                continue
            blk = parse_block(text, "dataset")
            if blk is None:
                unparsed += 1
                continue
            _s, _e, cur_items = blk
            cur = [i.strip("'\"") for i in cur_items]
            if cur == names:
                already_ok += 1
                continue
            to_fix += 1
            planned.append((p, cur, names))

    print(f"scanned {scanned} config files in {time.time()-t0:.0f}s")
    print(f"  recipe runs (processes_file set) : {recipe_runs}")
    print(f"  already correct / backfilled     : {already_ok}")
    print(f"  recipe file missing              : {missing_recipe}")
    print(f"  unparsed (odd yaml shape)        : {unparsed}")
    print(f"  STALE, would fix                 : {to_fix}")

    if planned:
        print("\n  examples:")
        for p, cur, names in planned[:3]:
            print(f"    {p.resolve().relative_to(REPO)}")
            print(f"      stale ({len(cur)}): {cur[:3]}{' ...' if len(cur) > 3 else ''}")
            print(f"      real  ({len(names)}): {names[:3]}{' ...' if len(names) > 3 else ''}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to write.")
        return 0
    if not planned:
        print("\nnothing to do.")
        return 0

    # restore point before touching anything
    stamp = subprocess.run(["date", "+%Y%m%d-%H%M%S"], capture_output=True, text=True).stdout.strip()
    backup = REPO / f"runs_config_backup_{stamp}.tar.gz"
    listing = REPO / f".backfill_filelist_{stamp}.txt"
    listing.write_text("\n".join(str(p.resolve().relative_to(REPO)) for p, _, _ in planned) + "\n")
    print(f"\nbacking up {len(planned)} files -> {backup.name}")
    r = subprocess.run(["tar", "czf", str(backup), "-C", str(REPO), "-T", str(listing)])
    if r.returncode != 0:
        print("BACKUP FAILED — refusing to write.", file=sys.stderr)
        return 1
    listing.unlink()

    written = 0
    for p, _cur, names in planned:
        text = p.read_text()
        blk = parse_block(text, "dataset")
        if blk is None:
            continue
        s, e, _ = blk
        m = re.search(r"^(?P<ind>[ \t]+)dataset:[ \t]*$", text[s:], re.M)
        indent = m.group("ind")
        new_block = f"{indent}dataset:\n" + "".join(f"{indent}- {n}\n" for n in names)
        new_block += f"{indent}dataset_backfilled: true\n"
        p.write_text(text[:s] + new_block + text[e:])
        written += 1

    print(f"rewrote {written} configs. Restore with:")
    print(f"  tar xzf {backup.name} -C {REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
