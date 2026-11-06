"""The random Lorentz augmentation drawn per event from a key, not from the global torch RNG (D290, 2026-11-06).

Each event's transform is a pure function of (augmentation seed, pool identity, index of the event in its pool): the
in-memory recipe path applies it to the whole pool at load, the memory-mapped path (data_mmap.py) to each batch as it
is gathered, and both get the same float32 momenta. So one data cache serves every seed: it stores the momenta in
their centre-of-mass frame, before the augmentation, and the seed enters only when a batch is read.

The distribution is lloca's rand_lorentz(std_eta=0.1) as experiment._boost_augment_momenta drew it before: a rotation
uniform over SO(3) (the same quaternion construction), times the rest-frame boost of the "four-momentum"
(gamma, beta) with beta ~ N(0, 0.1^2) per component (rand_boost; the library's truncation at 3 sigma is a no-op, its
clamp is not in place, so none is applied here either). The uniforms come from a counter-based hash (splitmix64) of
the key, the normals from Box-Muller on pairs of them. Every operation is elementwise in a fixed order, so an event's
transform and its transformed momenta do not depend on which other events share the array or the batch.
"""
import hashlib
import json

import numpy as np

STD_ETA = 0.1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_N_STREAMS = 9          # 3 uniforms for the rotation, 6 for the three Box-Muller normals of the boost


def _mix64(x):
    """splitmix64's finaliser on a uint64 array (wrapping arithmetic)"""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


HEAD_BYTES = 1 << 16


def pool_identity(path):
    """A pool file's content identity, independent of where it lies and when it was written: [recipe_id, size in
    bytes, sha256 of its first and last 64 KiB] (the .npy header carries the shape, so a pool cut or extended
    changes it). For a pool composed at load (mix_compose: its sidecar's derived_from.composed_at_load, rewritten in
    node-local /tmp by every job) the sidecar's sha256 of its rows stands for the content."""
    import os
    side_doc = None
    for side in (path + ".recipe.json", os.path.splitext(path)[0] + ".recipe.json"):
        if os.path.exists(side):
            try:
                side_doc = json.load(open(side))
            except (OSError, ValueError):
                pass
            break
    rid = (side_doc or {}).get("recipe_id")
    der = (side_doc or {}).get("derived_from") or {}
    size = os.path.getsize(path)
    if der.get("composed_at_load") and der.get("rows_sha256"):
        return [rid, int(size), "rows:" + der["rows_sha256"]]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(HEAD_BYTES))
        f.seek(max(0, size - HEAD_BYTES))
        h.update(f.read(HEAD_BYTES))
    return [rid, int(size), h.hexdigest()]


def pool_uid(identity):
    """A pool's 64-bit identity for the keys, from its content identity document (data_mmap.pool_identity)"""
    h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()
    return int.from_bytes(h[:8], "little")


def _uniforms(seed, uids, idx):
    """(n, _N_STREAMS) float64 uniforms in [0, 1) for the events (uids[i], idx[i]) under `seed`"""
    uids = np.asarray(uids, dtype=np.uint64)
    idx = np.asarray(idx, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        s = _mix64(np.array([int(seed) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64) + _GOLDEN)
        base = _mix64(s ^ uids)
        ev = _mix64(base + (idx + np.uint64(1)) * _GOLDEN)
        k = (np.arange(1, _N_STREAMS + 1, dtype=np.uint64) * np.uint64(0xD1B54A32D192ED03))
        h = _mix64(ev[:, None] + k[None, :])
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def trafo_components(seed, uids, idx):
    """The transforms rotation @ boost of the events (uids[i], idx[i]) as T[i][k], 4 x 4 contiguous (n,) float64
    arrays (component-major, so building and applying them runs over contiguous memory)"""
    u = _uniforms(seed, uids, idx)
    # rotation: lloca's rand_rotation (uniform quaternions)
    a, b, c = u[:, 0], u[:, 1], u[:, 2]
    q1 = np.sqrt(1 - a) * np.sin(2 * np.pi * b)
    q2 = np.sqrt(1 - a) * np.cos(2 * np.pi * b)
    q3 = np.sqrt(a) * np.sin(2 * np.pi * c)
    q0 = np.sqrt(a) * np.cos(2 * np.pi * c)
    R = [[1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
         [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
         [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]]
    # boost: lloca's rand_boost: beta ~ N(0, STD_ETA^2) per component, then restframe_boost((gamma, beta))
    z = np.sqrt(-2.0 * np.log1p(-u[:, 3:6])) * np.cos(2 * np.pi * u[:, 6:9])
    beta = [np.ascontiguousarray(z[:, j]) * STD_ETA for j in range(3)]
    gam0 = 1.0 / np.sqrt(np.maximum(1.0 - ((beta[0] ** 2 + beta[1] ** 2) + beta[2] ** 2), 1e-10))
    g0 = np.maximum(gam0, 1e-10)
    v = [bj / g0 for bj in beta]                                    # restframe_boost's beta = p / E
    v2 = (v[0] ** 2 + v[1] ** 2) + v[2] ** 2
    g = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-10))
    sc = (g - 1.0) / np.maximum(v2, 1e-10)
    B = [[g, -g * v[0], -g * v[1], -g * v[2]]]
    for i in range(3):
        B.append([-g * v[i]] + [(1.0 if i == j else 0.0) + sc * v[i] * v[j] for j in range(3)])
    T = [B[0]]
    for i in range(3):          # rotation (block 1:, 1:) @ boost, summed in a fixed order
        T.append([(R[i][0] * B[1][k] + R[i][1] * B[2][k]) + R[i][2] * B[3][k] for k in range(4)])
    return T


def event_trafos(seed, uids, idx):
    """(n, 4, 4) float64 Lorentz transforms rotation @ boost for the events (uids[i], idx[i])"""
    return np.stack([np.stack(row, axis=-1) for row in trafo_components(seed, uids, idx)], axis=-2)


def apply_components(T, mom):
    """T (trafo_components, each (n,), or already repeated per particle) applied to momenta (n, P, 4) or (m, 4);
    the sum over the inner index in a fixed order, in float64; returns float64"""
    p = np.asarray(mom)
    pc = [p[..., j].astype(np.float64) for j in range(4)]
    exp_ = (lambda t: t[:, None]) if p.ndim == 3 else (lambda t: t)
    out = np.empty(p.shape, dtype=np.float64)
    for i in range(4):
        t = [exp_(T[i][k]) for k in range(4)]
        out[..., i] = ((t[0] * pc[0] + t[1] * pc[1]) + t[2] * pc[2]) + t[3] * pc[3]
    return out


def augment(com, seed, uid, idx):
    """Centre-of-mass momenta `com` (n, P, 4) float32 of the events idx (n,) of pool `uid`, augmented: float32"""
    uids = np.full(len(idx), np.uint64(uid), dtype=np.uint64)
    return apply_components(trafo_components(seed, uids, idx), com).astype(np.float32)


def model_momenta(com, seed, uid, start, rs=None, mom_div=None):
    """What the network reads of the events start .. start + len(com) of pool `uid` (com: their centre-of-mass
    momenta, float32): augmented, in units of each event's sqrt(s_hat) when `rs` (their sqrt(s_hat), float64) is
    given, divided by the global momentum scale when `mom_div` is. Float32, the operations in the order both data
    paths apply them (data_mmap.MmapSource.gather does the same per particle)."""
    a = augment(com, seed, uid, np.arange(start, start + len(com), dtype=np.int64))
    if rs is not None:
        a = (a.astype(np.float64) / np.asarray(rs)[:, None, None]).astype(np.float32)
    return a if mom_div is None else a / mom_div


def model_momenta_chunks(com, seed, uid, rs=None, mom_div=None, chunk=1 << 16):
    """model_momenta over a whole pool in chunks of `chunk` events (bounded float64 temporaries), as (n, P, 4)
    float32 pieces in order"""
    for i in range(0, len(com), chunk):
        yield model_momenta(com[i:i + chunk], seed, uid, i, None if rs is None else rs[i:i + chunk], mom_div)
