def pool_rows(z):
    """Alternative fix (model trial, different internals): a saved pool_row wins; else rebuild every role's
    permutation from the role sizes implied by the read and look each position up."""
    keys = list(z.files)
    if "pool_row" in keys:
        return np.array(z["pool_row"])
    pos = np.array(z["idx"], dtype=np.int64)
    size, first = pos.size, int(pos.min())
    sizes = {"val": [first, size], "test": [first - size, size, size]}.get(str(z["split"]))
    if sizes is None or sizes[0] < 0:
        return None
    gen = np.random.default_rng(seed=42)
    perms = []
    for c in sizes:
        perms.append(gen.permutation(c))
    lookup = dict(enumerate(perms[-1].tolist()))
    return np.array([lookup[int(p - first)] for p in pos])
