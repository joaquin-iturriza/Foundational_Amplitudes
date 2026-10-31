"""catalog_ft follows the parent's coupling-monomial source in data.orders_from_diagrams."""
import json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import catalog_ft  # noqa: E402


def _parent(src):
    d = tempfile.mkdtemp()
    json.dump({"target_physics_orders_source": src}, open(os.path.join(d, "data_stats.json"), "w"))
    return d


def test_table():
    assert catalog_ft.parent_orders_flag(_parent("table"))[:2] == (False, "table")


def test_diagrams():
    assert catalog_ft.parent_orders_flag(_parent("diagrams_highest_alphas"))[:2] == (True, "diagrams_highest_alphas")


def test_alphas_only_refused():
    try:
        catalog_ft.parent_orders_flag(_parent("diagrams_alphas_only"))
    except ValueError:
        return
    raise AssertionError("alpha_s-only parent was not refused")


def test_command_flag():
    for flag, s in ((True, "true"), (False, "false")):
        cmd = catalog_ft.command("p", "AdamW", "r.yaml", 10, 5, "rd", "res", "e", orders_from_diagrams=flag)
        assert f"++data.orders_from_diagrams={s}" in cmd


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
