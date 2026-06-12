"""
Parse LHE files (LO from MadGraph or NLO from GoSam+Sherpa) into the .npy
format used by this project.

Output .npy layout (one row per event):
    [E0 px0 py0 pz0 | E1 px1 py1 pz1 | ... | pdg0 pdg1 ... | weight]

That is:
    - 4 * n_particles columns for four-momenta  (E, px, py, pz) per particle
    - n_particles columns for PDG IDs           (signed integers, stored as float64)
    - 1 column for the event weight             (cross-section weight in pb, can be negative for NLO)

ALL particles are kept in LHE order: incoming (status == -1) first, then outgoing
(status == 1).  Intermediate resonances (status == 2) are dropped.  This matches
the layout of the existing .npy datasets in this project (e.g. ee_aa_13000GeV_amplitudes.npy
has 4 particles: e-, e+, γ, γ with PDGs -11, 11, 22, 22).

For NLO files the weight can be negative (virtual/subtraction contributions).
The existing LO datasets use a log+standardisation preprocessing for the amplitude;
that preprocessing is applied at training time in experiment.py, so we store the
raw signed weight here.

Usage
-----
    # LO MadGraph output (all events have the same multiplicity):
    python parse_lhe.py --input proc_lo.lhe --output proc_lo.npy

    # NLO GoSam+Sherpa output (mixed 2->2 and 2->3 multiplicities):
    python parse_lhe.py --input eeuu_10000000.lhe --output eeuu_nlo_10M.npy

    # Separate mixed-multiplicity NLO into one file per multiplicity:
    python parse_lhe.py --input eeuu_10000000.lhe --output eeuu_nlo_10M.npy --split-multiplicity

    # Process only the first N events (useful for quick tests):
    python parse_lhe.py --input eeuu_10000000.lhe --output test.npy --max-events 10000
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# LHE parser
# ---------------------------------------------------------------------------

def _parse_lhe(path: str, max_events: int | None = None):
    """
    Generator that yields one dict per event:
        {
            'weight':    float,          # event weight in pb (can be negative for NLO)
            'pdg_ids':   list[int],      # ALL particle PDG IDs (incoming then outgoing)
            'momenta':   list[tuple],    # ALL (E, px, py, pz) per particle
            'n_total':   int,            # total number of particles (incoming + outgoing)
            'n_in':      int,            # number of incoming particles
            'n_out':     int,            # number of outgoing particles
        }

    Particles are kept in LHE file order: incoming (status==-1) first, then
    outgoing (status==1).  Intermediate resonances (status==2) are dropped.
    """
    in_event = False
    header_parsed = False
    particles = []
    weight = None
    n_events = 0

    opener = open  # plain text; extend here for .gz if needed

    with opener(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("<event"):
                in_event = True
                header_parsed = False
                particles = []
                weight = None
                continue

            if line.startswith("</event>"):
                in_event = False
                kept = [p for p in particles if p["status"] in (-1, 1)]
                if kept and weight is not None:
                    n_in  = sum(1 for p in kept if p["status"] == -1)
                    n_out = sum(1 for p in kept if p["status"] ==  1)
                    yield {
                        "weight":  weight,
                        "pdg_ids": [p["pdg"] for p in kept],
                        "momenta": [(p["E"], p["px"], p["py"], p["pz"]) for p in kept],
                        "n_total": len(kept),
                        "n_in":    n_in,
                        "n_out":   n_out,
                    }
                    n_events += 1
                    if max_events is not None and n_events >= max_events:
                        return
                continue

            if not in_event or line.startswith("#") or line.startswith("<"):
                continue

            cols = line.split()

            if not header_parsed:
                # Event header: n_particles  process_id  weight  sqrt(s)  alpha_s  alpha_ew
                weight = float(cols[2])
                header_parsed = True
                continue

            # Particle line:
            # pdg  status  mother1  mother2  color1  color2  px  py  pz  E  mass  hel  ?
            if len(cols) < 10:
                continue
            particles.append({
                "pdg":    int(cols[0]),
                "status": int(cols[1]),
                "px":     float(cols[6]),
                "py":     float(cols[7]),
                "pz":     float(cols[8]),
                "E":      float(cols[9]),
            })


# ---------------------------------------------------------------------------
# Build numpy array
# ---------------------------------------------------------------------------

def build_array(events: list[dict]) -> np.ndarray:
    """
    Stack events with the same n_total into a single array of shape
    (N, 4*n_total + n_total + 1) = (N, 5*n_total + 1).

    Column order: E0 px0 py0 pz0 ... En pxn pyn pzn  pdg0 ... pdgn  weight

    Particles appear in LHE order: incoming (status==-1) first, then outgoing
    (status==1), matching the layout of the existing .npy datasets.
    """
    n_total = events[0]["n_total"]
    assert all(e["n_total"] == n_total for e in events), \
        "All events must have the same total multiplicity to stack into one array."

    N = len(events)
    n_cols = 4 * n_total + n_total + 1
    arr = np.empty((N, n_cols), dtype=np.float64)

    for i, ev in enumerate(events):
        col = 0
        for (E, px, py, pz) in ev["momenta"]:
            arr[i, col:col+4] = [E, px, py, pz]
            col += 4
        for pdg in ev["pdg_ids"]:
            arr[i, col] = float(pdg)
            col += 1
        arr[i, col] = ev["weight"]

    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse LHE → .npy dataset")
    parser.add_argument("--input",  "-i", required=True,  help="Input LHE file")
    parser.add_argument("--output", "-o", required=True,  help="Output .npy file (or prefix if --split-multiplicity)")
    parser.add_argument("--split-multiplicity", action="store_true",
                        help="Write one .npy per multiplicity (e.g. for NLO files with real-emission events)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Stop after this many events (for quick tests)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: input file {args.input!r} not found.")

    print(f"Parsing {args.input} ...")

    # Bucket events by total multiplicity (n_in + n_out)
    buckets: dict[int, list[dict]] = defaultdict(list)
    n_total = 0
    n_negative = 0

    for ev in _parse_lhe(args.input, max_events=args.max_events):
        buckets[ev["n_total"]].append(ev)
        n_total += 1
        if ev["weight"] < 0:
            n_negative += 1
        if n_total % 500_000 == 0:
            print(f"  ... {n_total:,} events parsed", flush=True)

    print(f"Done. {n_total:,} events total.")
    print(f"  Negative weights: {n_negative:,} ({100*n_negative/n_total:.1f}%)")

    # Report multiplicity breakdown with in/out split
    mult_info = {}
    for n_tot, evs in sorted(buckets.items()):
        sample = evs[0]
        mult_info[n_tot] = {
            "count": len(evs),
            "n_in":  sample["n_in"],
            "n_out": sample["n_out"],
            "pdgs":  sample["pdg_ids"],
        }
        print(f"  {n_tot} particles ({sample['n_in']} in + {sample['n_out']} out): "
              f"{len(evs):,} events, PDGs (first event): {sample['pdg_ids']}")

    if args.split_multiplicity or len(buckets) > 1:
        if len(buckets) > 1 and not args.split_multiplicity:
            print("WARNING: multiple multiplicities detected but --split-multiplicity not set. "
                  "Writing one file per multiplicity anyway.")
        base, ext = os.path.splitext(args.output)
        for n_tot, events in sorted(buckets.items()):
            arr = build_array(events)
            info = mult_info[n_tot]
            suffix = f"_{info['n_in']}in{info['n_out']}out"
            out_path = f"{base}{suffix}{ext or '.npy'}"
            np.save(out_path, arr)
            print(f"  Saved {len(events):,} events ({n_tot} particles) -> {out_path}")
            print(f"    shape: {arr.shape}  weight range: [{arr[:,-1].min():.3e}, {arr[:,-1].max():.3e}]")
    else:
        n_tot = next(iter(buckets))
        events = buckets[n_tot]
        arr = build_array(events)
        out_path = args.output if args.output.endswith(".npy") else args.output + ".npy"
        np.save(out_path, arr)
        print(f"Saved {len(events):,} events -> {out_path}")
        print(f"  shape: {arr.shape}  weight range: [{arr[:,-1].min():.3e}, {arr[:,-1].max():.3e}]")


if __name__ == "__main__":
    main()
