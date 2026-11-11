"""Every repo module the KIT package's code imports is in the package (tools/kit_package.py public paths + EXTRA)."""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_package_imports_closed():
    from tools import kit_package as kp
    files = set()
    for p in kp.public_paths() + kp.EXTRA:
        files |= set(subprocess.run(["git", "ls-files", p], cwd=ROOT, capture_output=True, text=True).stdout.split())
    missing = set()
    for f in (f for f in files if f.endswith(".py")):
        try:
            tree = ast.parse(open(os.path.join(ROOT, f)).read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module + "." + a.name for a in n.names] + [n.module]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            for m in mods:
                c = m.replace(".", "/") + ".py"
                if os.path.exists(os.path.join(ROOT, c)) and c not in files:
                    missing.add(f"{c} <- {f}")
    assert not missing, sorted(missing)


if __name__ == "__main__":
    test_package_imports_closed()
    print("all ok")
