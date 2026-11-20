"""run.promote_final_model: the stage's checkpoints replace the pre-fit ones, a save_model_at_steps snapshot stays."""
import os
from run import promote_final_model


def test_keeps_step_snapshots(tmp_path):
    run, stage = tmp_path / "run", tmp_path / "run" / "sigma_stage"
    (run / "models").mkdir(parents=True)
    (stage / "models").mkdir(parents=True)
    for f in ("model_run0.pt", "model_run0_best.pt", "model_run0_step3458.pt"):
        (run / "models" / f).write_text("pre")
    for f in ("model_run0.pt", "model_run0_best.pt"):
        (stage / "models" / f).write_text("post")
    promote_final_model(str(run), str(stage))
    m = run / "models"
    assert sorted(os.listdir(m)) == ["model_run0.pt", "model_run0_best.pt", "model_run0_step3458.pt"]
    assert (m / "model_run0.pt").read_text() == "post" and (m / "model_run0_step3458.pt").read_text() == "pre"
    assert os.path.islink(stage / "models")
