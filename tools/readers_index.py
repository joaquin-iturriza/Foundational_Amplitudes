#!/usr/bin/env python3
"""One line per public function of analysis/common (all importable as `from analysis.common import <name>`):
name(signature), then its docstring's first sentence, cut at WIDTH characters. Read from
the source (ast), so it runs without the analysis environment; plan.py puts it in every step agent's prompt.
    python3 tools/readers_index.py"""
import ast, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDTH = 110                     # characters of the docstring's first sentence kept, so the step prompt stays short


def index(root=ROOT):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "analysis", "common", "*.py"))):
        for n in ast.parse(open(f).read()).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
                doc = " ".join((ast.get_docstring(n) or "").split("\n\n")[0].split())
                cut = min([i + 1 for i in (doc.find(". "), doc.find("; ")) if i >= 0] or [len(doc)])
                doc = doc[:cut].rstrip(";")
                doc = doc if len(doc) <= WIDTH else doc[:WIDTH - 3].rstrip() + "..."
                out.append("%s(%s)  %s" % (n.name, ast.unparse(n.args), doc))
    return out


if __name__ == "__main__":
    print("\n".join(index()))
    sys.exit(0)
