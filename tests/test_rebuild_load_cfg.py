"""tools/rebuild_run.load_cfg re-roots another site's project paths (a sweep trial copied lxplus -> CC, 2026-11-20)."""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_load_cfg_reroots_foreign_paths():
    import siteconf
    from tools.rebuild_run import load_cfg
    other = next(r for r in siteconf._OTHER_ROOTS if r != siteconf._S["project_dir"])
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write(f"data:\n  processes_file: {other}/runs/kit/recipes/x.yaml\n  path: '${{oc.env:DATA_DIR,/x}}'\n")
    cfg = load_cfg(d)
    assert cfg.data.processes_file == siteconf._S["project_dir"] + "/runs/kit/recipes/x.yaml"
    assert OmegaConf_raw(cfg) == "${oc.env:DATA_DIR,/x}"   # interpolations are kept, not resolved


def OmegaConf_raw(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=False)["data"]["path"]


if __name__ == "__main__":
    test_load_cfg_reroots_foreign_paths(); print("ok")
