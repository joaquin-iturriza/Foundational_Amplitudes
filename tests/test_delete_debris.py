"""tools/delete_debris.py on a fake SCRATCH and checkout (cc_disk, D550): deletes only the named smokes, format-2 memmap
keys no live job names, and old scratch run dirs; keeps logs, results, new scratch dirs and format-3 keys; skips a class
listed over 2x its expected size."""
import os, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "delete_debris.py")


def _tree(t):
    s, p, b = (os.path.join(t, x) for x in ("s", "p", "bin"))
    for d in ("amp_mmap_cache/oldkey", "amp_mmap_cache/newkey/d1", "zz_v3_smoke_Q_62382593", "zeroshot_old1",
              "zeroshot_new1", "zz_catft"):
        os.makedirs(os.path.join(s, d))
    os.makedirs(p); os.makedirs(b)
    open(os.path.join(s, "amp_mmap_cache/oldkey/meta.json"), "w").write('{"version": 2}')
    open(os.path.join(s, "amp_mmap_cache/newkey/d1/meta.json"), "w").write('{"key_doc": {"format": 3}}')
    open(os.path.join(s, "zz_v3_smoke_Q_62382593/f"), "wb").write(b"\0" * 1000)
    open(os.path.join(s, "zz_v3_smoke_Q_62382593.recipe.yaml"), "w").write("x")
    open(os.path.join(s, "zeroshot_old1/a"), "wb").write(b"\0" * 500)
    old = time.time() - 5 * 86400
    os.utime(os.path.join(s, "zeroshot_old1"), (old, old))
    log = os.path.join(p, "catalog_ft_11111111.out")      # only an end-of-run metric line: evidence, never deleted
    open(log, "w").write("[2026-11-01 00:00:00] MSE (prepd) val_ee_uu ee_uu: 0.01\n")
    os.utime(log, (old - 20 * 86400, old - 20 * 86400))
    for n, body in (("squeue", ""), ("scontrol", "echo StdOut=/nonexistent")):
        f = os.path.join(b, n)
        open(f, "w").write("#!/bin/sh\n%s\n" % body)
        os.chmod(f, 0o755)
    return s, p, b


def _run(s, p, b, *args):
    env = dict(os.environ, SCRATCH=s, PROJECT_DIR=p, PATH=b + os.pathsep + os.environ["PATH"], USER="x")
    return subprocess.run([sys.executable, TOOL, "--manifest", os.path.join(s, "..", "m.tsv"), *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def test_deletes_only_debris():
    with tempfile.TemporaryDirectory() as t:
        s, p, b = _tree(t)
        out = _run(s, p, b)
        assert "DEBRIS_DONE" in out
        left = {os.path.relpath(os.path.join(d, x), t) for d, ds, fs in os.walk(t) for x in ds + fs}
        for gone in ("s/zz_v3_smoke_Q_62382593", "s/zz_v3_smoke_Q_62382593.recipe.yaml", "s/amp_mmap_cache/oldkey",
                     "s/zeroshot_old1"):
            assert gone not in left, gone
        for kept in ("p/catalog_ft_11111111.out", "s/zeroshot_new1", "s/zz_catft", "s/amp_mmap_cache/newkey/d1/meta.json"):
            assert kept in left, kept


def test_class_over_twice_expected_is_skipped():
    with tempfile.TemporaryDirectory() as t:
        s, p, b = _tree(t)
        out = _run(s, p, b, "--expect-gib", "smoke=1e-9,mmap=4.8,scratch=2.2")
        assert "SKIPPED smoke" in out
        assert os.path.exists(os.path.join(s, "zz_v3_smoke_Q_62382593"))
        assert not os.path.exists(os.path.join(s, "zeroshot_old1"))


def test_smoke_models_old_unnamed_only():
    with tempfile.TemporaryDirectory() as t:
        s, p, b = _tree(t)
        old = time.time() - 5 * 86400
        runs = os.path.join(p, "runs", "zz_smoke")
        z = "zz_smoke" + "_"          # built here: a literal run name in this tracked file would count as named in the repo
        for n in (z + "old_1", z + "new_2", "zz_smoke_sweep_config_llmcmp_a_adamw_const_64771325"):
            os.makedirs(os.path.join(runs, n, "models"))
            for f in ("config.yaml", "out_0.log", "models/model_run0.pt", "best.ckpt"):
                open(os.path.join(runs, n, f), "wb").write(b"\0" * 100)
            if n != "zz_smoke_new_2":                    # the new one keeps today's mtimes
                for d, _, fs in os.walk(os.path.join(runs, n)):
                    for f in fs:
                        os.utime(os.path.join(d, f), (old, old))
        out = _run(s, p, b)
        assert "deleted smoke_models: 2 entries" in out, out
        o = os.path.join(runs, z + "old_1")
        assert not os.path.exists(os.path.join(o, "models/model_run0.pt")) and not os.path.exists(os.path.join(o, "best.ckpt"))
        assert os.path.exists(os.path.join(o, "config.yaml")) and os.path.exists(os.path.join(o, "out_0.log"))
        for n in (z + "new_2", "zz_smoke_sweep_config_llmcmp_a_adamw_const_64771325"):   # recent; named in the repo
            assert os.path.exists(os.path.join(runs, n, "models/model_run0.pt")), n


if __name__ == "__main__":
    test_deletes_only_debris(); test_class_over_twice_expected_is_skipped(); test_smoke_models_old_unnamed_only(); print("ok")
