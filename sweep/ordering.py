"""One submission order for every DyHPO search (D396: "submitting these searches is the same if we spread it across
sites or not"), used by sweep/drive.py (trials from the local machine to any site) and sweep/sweep_manager.py (the
classic path, state on one site); placement is the only thing that differs.

  wave_sizes      each search goes out in waves: its n_startup random trials, then the surrogate's guided trials in
                  two batches (D244: 8 trials, 3 start-up -> 3, 3, 2); wave k+1 is suggested only once waves 1..k of
                  that search have finished (drive.py counts them; sweep_manager chains them with afterany)
  dispatch_order  across searches: the trials wanting out now, longest trial first, round-robin across searches of
                  equal length, so under a submit cap the long trials start first and no search's wave goes before
                  the others'
"""


def wave_sizes(n_trials, n_startup, waves=None):
    """The search's wave sizes: `waves` when given (they must add up to n_trials), else the start-up wave and the rest
    in two guided waves."""
    if waves:
        waves = [int(w) for w in waves]
        if any(w <= 0 for w in waves) or sum(waves) != n_trials:
            raise ValueError("wave sizes %s must be positive and add up to n_trials=%d" % (waves, n_trials))
        return waves
    n_startup = max(0, min(int(n_startup), n_trials))
    rest = n_trials - n_startup
    return [w for w in (n_startup, (rest + 1) // 2, rest // 2) if w > 0]


def wave_of(pos, sizes):
    """The wave (0-based) holding the search's trial at position `pos`."""
    end = 0
    for k, w in enumerate(sizes):
        end += w
        if pos < end:
            return k
    return len(sizes) - 1


def dispatch_order(want, hours):
    """[key, ...] one entry per trial: `want` = {search: trials wanting out now}, `hours` = {search: trial length};
    longest first, round-robin across searches of equal length (ties by the order of `want`)."""
    out = []
    for h in sorted({hours[k] for k, n in want.items() if n > 0}, reverse=True):
        group = [[k] * n for k, n in want.items() if n > 0 and hours[k] == h]
        for r in range(max(len(g) for g in group)):
            out += [g[r] for g in group if r < len(g)]
    return out
