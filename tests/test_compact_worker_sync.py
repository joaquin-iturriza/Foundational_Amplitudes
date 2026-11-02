"""compact-worker.md carries a hand-made copy of the rule lines of both CLAUDE.md files; keep it in sync."""
from pathlib import Path

FA = Path(__file__).resolve().parents[1]
SOURCES = [FA / "CLAUDE.md", Path.home() / "work" / "CLAUDE.md"]
COPY = FA / ".claude" / "agents" / "compact-worker.md"


def _rules(path):
    return [l.rstrip("\n") for l in path.read_text().splitlines() if l.startswith("- ")]


def _key(line):
    return tuple(line[2:].split()[:5])


def test_compact_worker_rules_in_sync():
    src = {}
    for p in SOURCES:
        if not p.exists():
            continue
        for l in _rules(p):
            src.setdefault(_key(l), set()).add(l)
    stale, orphan = [], []
    for l in _rules(COPY):
        cands = src.get(_key(l))
        if cands is None:
            orphan.append(l)
        elif l not in cands:
            stale.append(l)
    assert not stale and not orphan, "stale (differs from CLAUDE.md):\n%s\nno counterpart:\n%s" % (
        "\n".join(stale), "\n".join(orphan))


if __name__ == "__main__":
    test_compact_worker_rules_in_sync()
    print("ok")
