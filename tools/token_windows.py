"""Coverage, model split and rolling-5h-window weighted tokens over ALL transcripts under ~/.claude/projects.
    python3 tools/token_windows.py [--out analysis/agent_redesign/token_windows.json]
Weights: input 1, output 5, cache write 1.25, cache read 0.1 (as tools/usage_report.py). One API message once per
(file, message id). A 'window' is the rolling 5 h ending at each message (sum of weighted tokens of messages in the
5 h before it); its max per period and the share of 5-hour windows (non-overlapping, aligned to first message) over a level.
"""
import argparse, glob, json, os, collections, bisect
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.expanduser("~/.claude/projects")
W = {"input_tokens": 1.0, "output_tokens": 5.0, "cache_creation_input_tokens": 1.25, "cache_read_input_tokens": 0.1}
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
B1, B2 = ts("2026-10-25T17:02:21Z"), ts("2026-10-30T11:35:10Z")
per = lambda t: "main_alone" if t < B1 else "step_agents" if t < B2 else "redesign"
a = argparse.ArgumentParser(); a.add_argument("--out", default="analysis/agent_redesign/token_windows.json"); a = a.parse_args()
cov, msgs = {}, []   # msgs: (t, model, weighted, is_sub, folder)
for d in sorted(os.listdir(P)):
    fs = glob.glob(os.path.join(P, d, "**", "*.jsonl"), recursive=True)
    ts_ = []; n_sub = 0
    for fp in fs:
        sub = "/subagents/" in fp; n_sub += sub; seen = set()
        for l in open(fp, errors="replace"):
            try: x = json.loads(l)
            except ValueError: continue
            if x.get("timestamp"): ts_.append(ts(x["timestamp"]))
            m = x.get("message") or {}
            if x.get("type") == "assistant" and m.get("usage") and m.get("id") not in seen:
                seen.add(m["id"]); u = m["usage"]
                msgs.append((ts(x["timestamp"]), m.get("model", "?"), sum(W[k] * (u.get(k) or 0) for k in W), sub, d))
    if fs:
        cov[d] = {"files": len(fs), "subagent_files": n_sub, "first": datetime.utcfromtimestamp(min(ts_)).isoformat() if ts_ else None,
                  "last": datetime.utcfromtimestamp(max(ts_)).isoformat() if ts_ else None}
msgs.sort()
for d in cov:
    mm = [m for m in msgs if m[4] == d]; cov[d]["api_messages"] = len(mm); cov[d]["weighted_M"] = round(sum(m[2] for m in mm) / 1e6, 2)
bymodel = collections.defaultdict(lambda: collections.defaultdict(float))
for t, mo, w, sub, d in msgs: bymodel[per(t)][mo] += w / 1e6
def windows(sel):
    t = [m[0] for m in sel]; c = [0.0]
    for m in sel: c.append(c[-1] + m[2])
    out = []
    for i, m in enumerate(sel):
        lo = bisect.bisect_left(t, m[0] - 5 * 3600); out.append((m[0], (c[i + 1] - c[lo]) / 1e6))
    return out
res = {"coverage": cov, "weighted_M_by_period_model": {p: {k: round(v, 2) for k, v in d.items()} for p, d in bymodel.items()}, "windows": {}}
# session since 2026-10-30 16:55:13 (calibration)
cal = sum(m[2] for m in msgs if m[0] >= ts("2026-10-30T16:55:13Z") and m[4] == "-home-joaquin-work-FA") / 1e6
res["since_2026-10-30T16:55:13_all_FA_transcripts_M"] = round(cal, 2)
for name, sel in (("all_models", msgs), ("opus", [m for m in msgs if "opus" in m[1]]), ("sonnet", [m for m in msgs if "sonnet" in m[1]])):
    w = windows(sel); R = {}
    for p in ("main_alone", "step_agents", "redesign"):
        v = [(t, x) for t, x in w if per(t) == p]
        if v:
            tm, mx = max(v, key=lambda a: a[1]); R[p] = {"max_5h_M": round(mx, 2), "at": datetime.utcfromtimestamp(tm).isoformat(),
                      "median_5h_M": round(sorted(x for _, x in v)[len(v) // 2], 2), "top_windows_M": []}
            # distinct top windows: greedy, separated by 5 h
            used = []
            for tt, x in sorted(v, key=lambda a: -a[1]):
                if all(abs(tt - u) > 5 * 3600 for u in used):
                    used.append(tt); R[p]["top_windows_M"].append([datetime.utcfromtimestamp(tt).isoformat(), round(x, 1)])
                if len(used) >= 8: break
    res["windows"][name] = R
json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("weighted_M_by_period_model", "since_2026-10-30T16:55:13_all_FA_transcripts_M", "windows")}, indent=1))
print(json.dumps({d: v for d, v in cov.items()}, indent=0)[:6000])
