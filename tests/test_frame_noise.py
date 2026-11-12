"""tools/rebuild_run.deterministic_frame_noise: lloca's degenerate-frame regularizers with noise that is a function of
each vector, so a frame depends on the particle alone (not on the RNG state or the batch); frames stay orthonormal."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.rebuild_run import deterministic_frame_noise, _hash_like  # noqa: E402


def test_frames_depend_on_the_particle_alone():
    torch.set_default_dtype(torch.float64)
    import lloca.utils.orthogonalize_3d as o3
    from lloca.utils.polar_decomposition import polar_decomposition
    deterministic_frame_noise()
    v = torch.zeros(5, 2, 3); v[:, 0, 2] = 1e-122; v[:, 1, 2] = 2e-122     # collinear rotation references
    v[3] = torch.tensor([[1., 0, 0], [0, 1., 0]]); v[4] = 0.0              # a regular pair; an exactly-zero pair
    a = o3.orthogonalize_3d(v)
    torch.manual_seed(3)
    b = o3.orthogonalize_3d(torch.cat([torch.randn(7, 2, 3) * 1e-130, v]))[7:]
    assert torch.equal(a, b), "the frame depends on the batch or the RNG"
    assert torch.allclose(a @ a.transpose(-1, -2), torch.eye(3).expand(5, 3, 3), atol=1e-12), "not orthonormal"
    p = torch.tensor([[2.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]])                  # a timelike and a lightlike boost vector
    refs = torch.zeros(2, 2, 4)
    t1 = polar_decomposition(p, refs); torch.manual_seed(9)
    t2 = polar_decomposition(torch.cat([torch.randn(3, 4).abs() + 5, p]), torch.cat([torch.zeros(3, 2, 4), refs]))[3:]
    assert torch.isfinite(t1).all() and torch.equal(t1, t2)
    c = o3.orthogonalize_3d(v * (1 + 1e-15))       # the degenerate references' bits moved (another batch's rounding)
    assert torch.equal(a[:3], c[:3]) and torch.equal(a[4], c[4]), "the frame follows the rounding of a degenerate reference"
    u = _hash_like(torch.randn(10000, 3), torch.arange(10000)[:, None])
    assert (u.abs() >= 1).all() and (u.abs() <= 2).all() and abs(float(u.mean())) < 0.05


if __name__ == "__main__":
    test_frames_depend_on_the_particle_alone(); print("ok")
