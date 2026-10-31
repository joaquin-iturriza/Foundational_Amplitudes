#!/usr/bin/env python3
"""Claude usage from every local transcript, priced per call at list price, summed per 5-hour block.

  extract     read every transcript under ~/.claude/projects (main sessions */*.jsonl and subagents
              */*/subagents/agent-*.jsonl), one record per API response deduplicated by message id over all
              files (a call can be logged in a subagent file and as a progress record of its parent), into
              ~/.cache/fa_usage/calls.jsonl
  inventory   per transcript folder: sessions, subagent files, calls, first and last call, and whether
              tools/usage_report.py read it
  calibrate   cost (and the old weighting) between Joaquin's usage-meter readings, per counting scope
  blocks      cost per 5-hour block, per period, model and source; percent at a cost-per-percent

Cost per call = list price per MTok (PRICES, the claude-api skill's model table, cached 2026-10-06; cache write
1.25x input for the 5-minute TTL and 2x for the 1-hour TTL, cache read per model, shared/prompt-caching.md).
Old weighting = input + 5 output + 1.25 cache write + 0.1 cache read (the main session's 2026-10-30 pilot).
"""
import argparse, glob, json, os, sys
from collections import defaultdict

PROJ = os.path.expanduser("~/.claude/projects")
CACHE = os.path.expanduser("~/.cache/fa_usage/calls.jsonl")
LIMITS = os.path.expanduser("~/.cache/fa_usage/limits.jsonl")
HIT = __import__("re").compile(r"You've hit your (5-hour|session|weekly) limit[^\"\\]{0,12}resets ([0-9:]+\s*[ap]m)")
THIS = "71602535-0955-42ad-a58f-23dc320798dc"       # the FA main session of 2026-10-22..
P1, P2 = "2026-10-25T17:02", "2026-10-30T11:35"     # step agents in; redesign live
PERIODS = ("main alone", "step agents", "redesign")
# $/MTok: input, output, cache read. Source: claude-api skill (2.1.312) model table and shared/models.md, cached
# 2026-10-06. Cache writes: 1.25x input (5m), 2x input (1h).
PRICES = [("claude-opus-5-5", 4.0, 20.0, 0.20), ("claude-opus-5", 5.0, 25.0, 0.50),
          ("claude-opus-4", 5.0, 25.0, 0.50), ("claude-sonnet-5-5", 2.0, 10.0, 0.20),
          ("claude-sonnet-5", 2.0, 10.0, 0.20), ("claude-sonnet-4-6", 3.0, 15.0, 0.30),
          ("claude-sonnet-4", 3.0, 15.0, 0.30), ("claude-haiku-5-5", 0.10, 0.50, 0.01),
          ("claude-haiku-4", 1.0, 5.0, 0.10), ("claude-fable-5-1", 10.0, 50.0, 0.25),
          ("claude-fable-5", 10.0, 50.0, 1.0)]
READINGS = [("2026-10-30T16:54:25", 5), ("2026-10-30T16:59:04", 7), ("2026-10-30T19:37:53", 28)]


def period(t):
    return PERIODS[0] if t < P1 else PERIODS[1] if t < P2 else PERIODS[2]


def price(model):
    for pre, i, o, r in PRICES:
        if (model or "").startswith(pre):
            return i, o, r
    return None


def cost(c):
    """$ of one call at list price; None for a model without a price (listed apart)."""
    p = price(c["model"])
    if p is None:
        return None
    i, o, r = p
    return (c["in"] * i + c["out"] * o + c["cw5"] * 1.25 * i + c["cw1h"] * 2.0 * i + c["cwu"] * 1.25 * i
            + c["cr"] * r) / 1e6


def old_weight(c):
    return c["in"] + 5 * c["out"] + 1.25 * (c["cw5"] + c["cw1h"] + c["cwu"]) + 0.1 * c["cr"]


def source(folder, f):
    if folder == "-home-joaquin-work-FA":
        if os.path.basename(f) == THIS + ".jsonl":
            return "this session"
        if "/" + THIS + "/subagents/" in f:
            return "this session's agents"
        return "other FA sessions"
    if folder.startswith("-home-joaquin-work-FA-"):
        return "FA worktrees and replays"
    if folder.startswith("-tmp-"):
        return "claude -p test sessions"
    return "other projects"


def _messages(d):
    """The API responses in one transcript record: its own message, or a subagent's nested in a progress record."""
    m = d.get("message")
    if isinstance(m, dict) and m.get("usage") and m.get("id"):
        yield m, d.get("timestamp")
    data = d.get("data")
    if isinstance(data, dict):
        mm = data.get("message")
        if isinstance(mm, dict):
            inner = mm.get("message")
            if isinstance(inner, dict) and inner.get("usage") and inner.get("id"):
                yield inner, mm.get("timestamp") or d.get("timestamp")


def files():
    return sorted(glob.glob(PROJ + "/*/*.jsonl")) + sorted(glob.glob(PROJ + "/*/*/subagents/agent-*.jsonl"))


def extract(out=CACHE):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seen, n, hits = {}, 0, {}
    for f in files():
        folder = f[len(PROJ) + 1:].split("/")[0]
        agent = os.path.basename(f)[6:-6] if "/subagents/" in f else None
        session = f.split("/")[-3] if agent else os.path.basename(f)[:-6]
        for line in open(f, errors="replace"):
            if "hit your" in line:
                h = HIT.search(line)
                if h:
                    try:
                        dd = json.loads(line)
                    except ValueError:
                        dd = {}
                    t = dd.get("timestamp") or ""
                    # the API's own refusal only (a quoted copy in a notification or a tool output is not one)
                    if t and dd.get("type") == "assistant" and dd.get("isApiErrorMessage") and dd.get("error") == "rate_limit":
                        hits.setdefault(t[:19], {"ts": t[:19], "limit": h.group(1), "resets": h.group(2),
                                                 "folder": folder})
            if '"usage"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            for m, ts in _messages(d):
                u = m["usage"]
                cc = u.get("cache_creation") or {}
                cw = u.get("cache_creation_input_tokens") or 0
                cw5, cw1h = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
                rec = {"id": m["id"], "ts": (ts or "")[:19], "model": m.get("model"), "in": u.get("input_tokens") or 0,
                       "out": u.get("output_tokens") or 0, "cw5": cw5, "cw1h": cw1h,
                       "cwu": max(cw - cw5 - cw1h, 0), "cr": u.get("cache_read_input_tokens") or 0,
                       "folder": folder, "session": session, "agent": agent, "source": source(folder, f)}
                old = seen.get(m["id"])
                # the subagent's own file wins over its parent's progress copy (it names the agent)
                if old is None or (rec["agent"] and not old["agent"]):
                    seen[m["id"]] = rec
    with open(out, "w") as fh:
        for r in sorted(seen.values(), key=lambda r: r["ts"]):
            if r["model"] == "<synthetic>":
                continue
            fh.write(json.dumps(r) + "\n")
            n += 1
    with open(LIMITS, "w") as fh:
        for h in sorted(hits.values(), key=lambda h: h["ts"]):
            fh.write(json.dumps(h) + "\n")
    print("%d calls from %d files -> %s; %d limit messages -> %s" % (n, len(files()), out, len(hits), LIMITS))


def load(path=CACHE):
    return [json.loads(l) for l in open(path)]


def inventory(calls):
    by = defaultdict(lambda: {"sessions": set(), "agents": set(), "calls": 0, "first": "9", "last": ""})
    for c in calls:
        b = by[c["folder"]]
        b["sessions"].add(c["session"]); b["calls"] += 1
        if c["agent"]:
            b["agents"].add(c["agent"])
        b["first"], b["last"] = min(b["first"], c["ts"]), max(b["last"], c["ts"])
    # usage_report.py @5b14ab22 reads -home-joaquin-work-FA only (main files and subagents)
    rows = [{"folder": k, "sessions": len(v["sessions"]), "agents": len(v["agents"]), "calls": v["calls"],
             "first": v["first"], "last": v["last"], "in_usage_report": k == "-home-joaquin-work-FA"}
            for k, v in sorted(by.items())]
    return rows


def window_starts(calls, floor=False):
    """5-hour window starts over all calls of the account: a window starts at the first call at or after the previous
    window's end (floor=True: that call's time rounded down to the hour)."""
    from datetime import datetime, timedelta
    out, end = [], None
    for c in calls:
        if not c["ts"]:
            continue
        t = datetime.fromisoformat(c["ts"])
        if end is None or t >= end:
            st = t.replace(minute=0, second=0) if floor else t
            out.append(st)
            end = st + timedelta(hours=5)
    return out


def _window_of(starts, t):
    import bisect
    i = bisect.bisect_right(starts, t) - 1
    return starts[i] if i >= 0 else None


def limit_hits():
    """First API rate-limit refusal per reset time: (UTC time, reset as UTC datetime, limit kind)."""
    from datetime import datetime, timedelta
    out, seen = [], set()
    for l in open(LIMITS):
        h = json.loads(l)
        t = datetime.fromisoformat(h["ts"])
        hh, ap = h["resets"].lower().replace(" ", "")[:-2], h["resets"].lower()[-2:]
        hr, mn = (hh.split(":") + ["0"])[:2]
        hr = int(hr) % 12 + (12 if ap == "pm" else 0)
        off = 2 if t < datetime(2026, 10, 25, 1) else 1          # Europe/Paris: CEST until 2026-10-25 01:00 UTC
        r = t.replace(hour=0, minute=0, second=0) + timedelta(hours=hr - off, minutes=int(mn))
        while r <= t:
            r += timedelta(days=1)
        if r in seen:
            continue
        seen.add(r)
        out.append({"hit": t, "reset_utc": r, "limit": h["limit"], "folder": h["folder"]})
    return out


SCOPES = {"this session and its agents": lambda c: c["source"] in ("this session", "this session's agents"),
          "every call of the account": lambda c: True}


def calibrate(calls):
    """Window convention against the reset times; cost per percent from the 23-point step; the other readings and
    every 100% hit predicted from it, per scope; the same with the old weighting."""
    from datetime import datetime
    conv = {}
    for name, floor in (("first call", False), ("floored hour", True)):
        st = window_starts(calls, floor)
        rows = []
        for h in limit_hits():
            w = _window_of(st, h["hit"])
            from datetime import timedelta
            rows.append({"hit": h["hit"].isoformat(), "limit": h["limit"], "reset_utc": h["reset_utc"].isoformat(),
                         "window_start": w and w.isoformat(),
                         "predicted_end": w and (w + timedelta(hours=5)).isoformat(),
                         "end_minus_reset_min": w and round(((w + timedelta(hours=5)) - h["reset_utc"]).total_seconds() / 60)})
        conv[name] = {"starts": len(st), "hits": rows,
                      "max_abs_end_error_min": max(abs(r["end_minus_reset_min"]) for r in rows if r["window_start"]
                                                   and r["hit"] >= "2026-10")}
    best = min(conv, key=lambda k: conv[k]["max_abs_end_error_min"])
    starts = window_starts(calls, best == "floored hour")
    t0, pct0 = datetime.fromisoformat(READINGS[0][0]), READINGS[0][1]
    t1, pct1 = datetime.fromisoformat(READINGS[2][0]), READINGS[2][1]
    out = {"window_convention": conv, "used": best, "scopes": {}}
    for sname, keep in SCOPES.items():
        sel = [c for c in calls if keep(c) and c["ts"]]
        T = [datetime.fromisoformat(c["ts"]) for c in sel]

        def tot(lo, hi, f):
            return sum(f(c) or 0 for c, t in zip(sel, T) if lo <= t < hi)
        res = {}
        for mname, f in (("usd list price", cost), ("old weighting", old_weight)):
            per = tot(t0, t1, f) / (pct1 - pct0)
            ws = _window_of(starts, t0)
            checks = [{"what": "%d%% at %s from its window start %s" % (pct0, READINGS[0][0], ws.isoformat()),
                       "predicted_pct": round(tot(ws, t0, f) / per, 1), "read_pct": pct0}]
            tm = datetime.fromisoformat(READINGS[1][0])
            checks.append({"what": "%d%% at %s (2-point step, +-50%%)" % (READINGS[1][1], READINGS[1][0]),
                           "predicted_pct": round(pct0 + tot(t0, tm, f) / per, 1), "read_pct": READINGS[1][1]})
            for h in limit_hits():
                w = _window_of(starts, h["hit"])
                if w is None:
                    continue
                checks.append({"what": "100%% at the %s-limit refusal %s (window from %s)" % (
                    h["limit"], h["hit"].isoformat(), w.isoformat()),
                    "predicted_pct": round(tot(w, h["hit"], f) / per, 1), "read_pct": 100})
            for ch in checks:
                ch["ratio"] = round(ch["predicted_pct"] / ch["read_pct"], 3)
            res[mname] = {"per_pct": round(per, 4) if mname.startswith("usd") else round(per),
                          "per_pct_range_1pt": [round(tot(t0, t1, f) / (pct1 - pct0 + 1), 4),
                                                round(tot(t0, t1, f) / (pct1 - pct0 - 1), 4)],
                          "checks": checks,
                          "pass_20pct": all(abs(c["ratio"] - 1) <= 0.2 for c in checks if "2-point" not in c["what"])}
        out["scopes"][sname] = res
    return out


def blocks(calls, keep=lambda c: True):
    """Cost per 5-hour window (first-call convention, chained over every call of the account), split by period of the
    call, model and source; the window's limit refusals, if any."""
    from datetime import datetime, timedelta
    starts = window_starts(calls)
    hits = limit_hits()
    out = {}
    for c in calls:
        if not c["ts"] or not keep(c):
            continue
        w = _window_of(starts, datetime.fromisoformat(c["ts"]))
        b = out.setdefault(w, {"start": w.isoformat(), "calls": 0, "usd": 0.0, "old_weight": 0.0, "first": c["ts"],
                               "last": c["ts"], "by_model": defaultdict(float), "by_source": defaultdict(float),
                               "by_period": defaultdict(float), "cache_read_usd": 0.0, "unpriced_calls": 0})
        u = cost(c)
        b["calls"] += 1; b["last"] = c["ts"]; b["old_weight"] += old_weight(c)
        if u is None:
            b["unpriced_calls"] += 1
            continue
        b["usd"] += u
        b["cache_read_usd"] += c["cr"] * price(c["model"])[2] / 1e6
        b["by_model"][c["model"]] += u; b["by_source"][c["source"]] += u; b["by_period"][period(c["ts"])] += u
    res = []
    for w, b in sorted(out.items()):
        for k in ("by_model", "by_source", "by_period"):
            b[k] = {m: round(v, 2) for m, v in sorted(b[k].items(), key=lambda x: -x[1])}
        b["usd"], b["old_weight"], b["cache_read_usd"] = round(b["usd"], 2), round(b["old_weight"]), round(b["cache_read_usd"], 2)
        b["active_h"] = round((datetime.fromisoformat(b["last"]) - w).total_seconds() / 3600, 2)
        b["period"] = max(b["by_period"], key=b["by_period"].get) if b["by_period"] else period(b["start"])
        b["limit_hit"] = [h["hit"].isoformat() for h in hits if w <= h["hit"] < w + timedelta(hours=5)]
        b["usd_at_hit"] = None
        if b["limit_hit"]:
            t = b["limit_hit"][0]
            b["usd_at_hit"] = round(sum(cost(c) or 0 for c in calls if c["ts"] and w.isoformat() <= c["ts"] < t), 2)
        res.append(b)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "inventory", "calibrate", "blocks", "report"])
    ap.add_argument("--since", default="2026-10-01")
    ap.add_argument("--usd-per-pct", type=float, help="cost per meter percent (from calibrate)")
    ap.add_argument("--out", help="write the report JSON here")
    a = ap.parse_args()
    if a.cmd == "extract":
        return extract()
    calls = [c for c in load() if c["ts"] >= a.since or a.cmd == "inventory"]
    if a.cmd == "inventory":
        r = inventory(calls)
    elif a.cmd == "calibrate":
        r = calibrate(calls)
    elif a.cmd == "blocks":
        r = blocks(calls)
    else:
        bl = blocks(calls)
        if a.usd_per_pct:
            for b in bl:
                b["pct"] = round(b["usd"] / a.usd_per_pct)
        models = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        for c in calls:
            m = models[period(c["ts"])][c["model"]]
            m[0] += 1; m[1] += cost(c) or 0
        r = {"prices": {"source": "claude-api skill 2.1.312, model table cached 2026-10-06; cache write 1.25x (5m) / "
                                  "2x (1h) input, shared/prompt-caching.md",
                        "per_mtok_in_out_read": {p[0]: p[1:] for p in PRICES}},
             "inventory": inventory(load()), "calibration": calibrate(calls), "usd_per_pct": a.usd_per_pct,
             "models_per_period": {p: {m: {"calls": v[0], "usd": round(v[1], 2)} for m, v in d.items()}
                                   for p, d in models.items()},
             "blocks": bl}
        hit_usd = [b["usd_at_hit"] for b in bl if b["usd_at_hit"] and b["start"] >= "2026-10-01"]
        lo = min(hit_usd) if hit_usd else None
        summ = {}
        for p in PERIODS:
            bs = [b for b in bl if b["period"] == p and b["start"] >= a.since and b["usd"] >= 1.0]
            us = sorted(b["usd"] for b in bs)
            summ[p] = {"windows_with_use": len(bs), "median_usd": us and us[len(us) // 2], "max_usd": us and us[-1],
                       "windows_at_or_above_lowest_hit_usd": lo and sum(u >= lo for u in us),
                       "windows_at_or_above_75pct_of_it": lo and sum(u >= 0.75 * lo for u in us),
                       "limit_hits": sum(bool(b["limit_hit"]) for b in bs),
                       "usd_per_active_hour": round(sum(us) / max(1e-9, sum(b["active_h"] for b in bs)), 2)}
        r["lowest_usd_at_a_limit_hit_since_oct"] = lo
        r["usd_at_limit_hits"] = hit_usd
        r["period_summary"] = summ
    s = json.dumps(r, indent=1, default=list)
    if a.out:
        open(a.out, "w").write(s + "\n")
        print("wrote", a.out)
    else:
        print(s)


if __name__ == "__main__":
    sys.exit(main())
