# AFS-side stub: runs the real script from this checkout (wherever it lives).
import os
import runpy
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
runpy.run_path(os.path.join(_root, "sweep/generate_sweep.py"), run_name="__main__")
