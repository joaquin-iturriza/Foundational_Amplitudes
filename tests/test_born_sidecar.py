"""The one-loop pipeline keeps MadLoop's Born per event (tools/nlo_virtual_pipeline.py: born_path, born_of_pool), with
MadLoop replaced by a deterministic fake of its Laurent array: generation writes <pool>.born.npy row-aligned and leaves
the pool's target as before ((c0 + shift) x Born), and the re-evaluation of an existing pool recomputes the stored
target exactly and returns the same Born, c0 and shift."""
import os
import sys

import numpy as np
import pytest

ROOT = os.environ.get("FA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nlo_virtual_pipeline as NV  # noqa: E402
import nlo_madloop as ML  # noqa: E402
import nlo_pole_check as PC  # noqa: E402


def fake_module(process):
    cfg = NV.VIRT_PROCESSES[process]
    c2 = PC.double_pole(cfg["pdg_ids"], [0.0, 0.0] + list(cfg["m_finals"]))

    def get_me_full(P, alphas, s, _):
        mom = np.asarray(P).T
        born = 1.0 + float(np.abs(mom[2]).sum()) / 100.0          # positive, event-dependent
        c0 = np.cos(float(mom[3, 1]))                              # event-dependent finite coefficient
        a = alphas / (2 * np.pi)
        return np.array([born, c0 * born * a, 0.0, c2 * born * a]), 0
    return get_me_full


@pytest.fixture
def patched(monkeypatch, tmp_path):
    def setup(process):
        monkeypatch.setattr(NV, "virt_standalone_dir", lambda p: str(tmp_path))
        monkeypatch.setattr(NV, "restore_stale_card", lambda *a, **k: None)
        monkeypatch.setattr(NV, "massive_beam_params", lambda *a, **k: None)
        monkeypatch.setattr(NV, "find_p0", lambda sa: str(tmp_path))
        monkeypatch.setattr(ML, "load", lambda p0: fake_module(process))
    return setup


@pytest.mark.parametrize("process", ["ee_uu", "ee_ttbar"])
def test_generation_keeps_born_and_reeval_reproduces(patched, tmp_path, process):
    patched(process)
    out = str(tmp_path / f"{process}_pool.npy")
    lo = 400.0 if process == "ee_ttbar" else 50.0
    NV.generate_virt_dataset(process, lo, 1000.0, 30, out, seed=3)
    pool = np.load(out)
    born = np.load(NV.born_path(out))
    assert born.shape == (30,) and np.all(born > 0)
    s = NV.born_of_pool(process, out, out=str(tmp_path / "re.npz"))
    assert s["n_unusable"] == 0 and s["n_rel_above_1e-6"] == 0 and s["max_rel"] < 1e-12
    z = np.load(tmp_path / "re.npz")
    np.testing.assert_array_equal(z["born"], born)
    np.testing.assert_array_equal(z["stored"], pool[:, -1])
    # the stored target is (c0 + shift) x Born, so the Born-relative target is log|c0 + shift|
    np.testing.assert_allclose((z["c0"] + z["shift"]) * z["born"], pool[:, -1], rtol=1e-12)
    assert (np.abs(z["shift"]).max() > 0) == (process == "ee_ttbar")
    # a subset of rows: the same values on those rows
    sub = NV.born_of_pool(process, out, n_events=10, seed=1, out=str(tmp_path / "sub.npz"))
    zs = np.load(tmp_path / "sub.npz")
    assert sub["n_evaluated"] == 10 and len(set(zs["idx"])) == 10
    np.testing.assert_array_equal(zs["born"], born[zs["idx"]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
