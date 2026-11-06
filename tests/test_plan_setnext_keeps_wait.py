"""`plan set-next` rewriting a step that waits on runs keeps the wait when the new text names those runs, and says so
when it drops one (2026-11-06: warm_start step 11's wait on its two re-runs was dropped silently by a set-next that
restated the step with their ids; both runs ended with no RUN_DONE)."""
import argparse, os, sys, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = "FA-ccin2p3-12cb7292 FA-ccin2p3-37dd5af5"


def load(tmp):
    os.environ.update(PLAN_TEST="1", PLAN_ROOT=str(tmp), PLAN_OWNER_MODE="1")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib, plan
    return importlib.reload(plan)


def put(tmp):
    e = {"id": "ent", "status": "running", "next": {11: "Step 11: the sweep"}, "step_waits": {"11": IDS}}
    os.makedirs(os.path.join(str(tmp), "plan"), exist_ok=True)
    with open(os.path.join(str(tmp), "plan", "plan.yaml"), "w") as f:
        yaml.safe_dump({"entries": [e], "migrated_at": "2026-10-01T00:00:00+00:00"}, f)


def setnext(plan, text):
    plan.cmd_set_next(argparse.Namespace(id="ent", text=text, also=None, next=None, feeds=None,
                                           seq=[("feeds", "whether the change becomes the recipe"),
                                                ("if", "BETTER: adopt it"), ("if", "NOT BETTER: keep the default")]))
    d, _ = plan.load()
    return d["entries"][0].get("step_waits") or {}


def test_restated_with_full_ids_keeps(tmp_path, capsys):
    plan = load(tmp_path); put(tmp_path)
    assert setnext(plan, "Step 11: re-runs " + IDS + "; read them when both end") == {"11": IDS}
    assert "keeps its wait" in capsys.readouterr().out


def test_restated_with_bare_suffixes_keeps(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    assert setnext(plan, "Step 11: re-runs 12cb7292 and 37dd5af5") == {"11": IDS}


def test_new_text_without_ids_drops_and_says(tmp_path, capsys):
    plan = load(tmp_path); put(tmp_path)
    assert setnext(plan, "Step 11: something else entirely") == {}
    assert "dropped" in capsys.readouterr().out


def test_one_id_missing_drops(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    assert setnext(plan, "Step 11: only 12cb7292") == {}
