"""A HTCondor job's flavour is the shortest whose limit covers its wall time, never the site's default
('tomorrow', 24 h): the 4 h and 12 h long-run trials, a 28 h one, and the D-HH:MM:SS form."""
import importlib.machinery, importlib.util, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep"))
from generate_sweep import condor_flavour

CASES = {"04:00:00": "workday", "12:00:00": "tomorrow", "28:00:00": "testmatch", "1-04:00:00": "testmatch",
         "00:15:00": "espresso", "00:50:00": "microcentury", "100:00:00": "nextweek"}


def test_generate_sweep():
    for t, f in CASES.items():
        assert condor_flavour(t) == f, (t, condor_flavour(t))


def _site_tool():
    p = os.path.expanduser("~/.local/bin/site")
    if not os.path.exists(p):
        return None
    loader = importlib.machinery.SourceFileLoader("site_tool", p)
    spec = importlib.util.spec_from_loader("site_tool", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def test_site_tool():
    m = _site_tool()
    if m is None:
        return
    for t, f in CASES.items():
        sub = m.condor_sub({"--time": t}, "/afs/x/job.sh")[1]
        assert '+JobFlavour = "%s"' % f in sub, (t, sub)



def test_gpu_min_mem():
    m = _site_tool()
    if m is None:
        return
    sub = m.condor_sub({"--time": "12:00:00", "--gres": "gpu:1", "--gpu-min-mem": "24G"}, "/afs/x/job.sh")[1]
    assert "gpus_minimum_memory = 24576" in sub and '+JobFlavour = "tomorrow"' in sub, sub
    assert "gpus_minimum_memory" not in m.condor_sub({"--gres": "gpu:1"}, "/afs/x/job.sh")[1]


if __name__ == "__main__":
    test_generate_sweep(); test_site_tool(); test_gpu_min_mem(); print("ok")
