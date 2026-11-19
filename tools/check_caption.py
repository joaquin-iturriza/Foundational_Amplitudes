#!/usr/bin/env python3
"""Run send_check's checks on one figure and caption as if sent: tools/check_caption.py <png> "<caption>".
Prints OK, or each refusal; exit 1 on refusal."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".claude", "hooks"))
import send_check as sc


def refusals(path, caption):
    path = os.path.abspath(path)
    ti = {"file_path": path, "caption": caption}
    bad = [l for p in sc.sent_files(ti) for l in sc.check(p, sc.glossary())]
    bad += sc.axes_findings(ti, os.getcwd())
    bad += sc.legend_findings(ti, os.getcwd())[0]
    return bad


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: check_caption.py <png> "<caption>"')
    bad = refusals(*sys.argv[1:])
    print("\n".join(bad) if bad else "OK")
    sys.exit(1 if bad else 0)
