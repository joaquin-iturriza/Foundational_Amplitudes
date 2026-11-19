import importlib.machinery, importlib.util, os, subprocess, sys, types
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_free_floor_alarms(tmp_path, monkeypatch, capsys):
    """512 GB fs with 40 GB free (92% used is not needed: 8% used would pass --warn) -> DISK line, exit 1."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import disk_watch as dw
    monkeypatch.setattr(dw, "CACHE", str(tmp_path / "c.json"))
    monkeypatch.setattr(dw, "FLAG", str(tmp_path / "flag"))
    monkeypatch.setattr(dw, "du_top", lambda a, b: ([], False))
    monkeypatch.setattr(dw, "usage", lambda p: (400e9, 512e9, 80))
    monkeypatch.setattr(dw, "free_gb", lambda p: 40.0)
    monkeypatch.setattr(sys, "argv", ["disk_watch"])
    try:
        dw.main()
        assert False, "no alarm"
    except SystemExit as e:
        assert e.code == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith("at 80% (40 GB free)")
    monkeypatch.setattr(dw, "free_gb", lambda p: 100.0)
    dw.main()   # 80% used, 100 GB free: no alarm


def test_disk_alarm_never_a_repeat():
    import sqlite3
    site = types.ModuleType("site_tool")
    site.__file__ = os.path.expanduser("~/.local/bin/site")
    src = open(site.__file__).read().split("\nif __name__")[0]
    exec(compile(src, site.__file__, "exec"), site.__dict__)
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, project, kind, key, msg)")
    out = "x\nDISK /sps/lpnhe at 92% (40 GB free)\n"
    assert not site.maintain_repeat(c, "FA", "cmd", out)
    assert not site.maintain_repeat(c, "FA", "cmd", out)
    assert site.maintain_repeat(c, "FA", "cmd", "boom 1\n") is False
    assert site.maintain_repeat(c, "FA", "cmd", "boom 2\n") is True


def test_alarm_only_on_change(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import disk_watch as dw
    for k, v in dict(CACHE=str(tmp_path / "c"), FLAG=str(tmp_path / "f"), STATE=str(tmp_path / "s")).items():
        monkeypatch.setattr(dw, k, v)
    monkeypatch.setattr(dw, "du_top", lambda a, b: ([], False))
    monkeypatch.setattr(dw, "usage", lambda p: (400e9, 512e9, 93))
    monkeypatch.setattr(sys, "argv", ["disk_watch"])

    def run(free):
        monkeypatch.setattr(dw, "free_gb", lambda p: free)
        try:
            dw.main()
            return 0
        except SystemExit as e:
            return e.code
    assert run(42.0) == 1      # first crossing
    assert run(41.0) == 0      # unchanged
    assert run(37.5) == 0      # -4.5 GB since the alert
    assert run(37.0) == 1      # -5 GB
    assert run(37.0) == 0
