#!/usr/bin/env python3
"""Can a lab LLM take the main session's routing wakes (D551: what else to use the lab's LLMs on)? Real wakes of the
main session that only routed or recorded (main_turns: trigger waiter, agent_done or handback; did orchestration or
none) are replayed to deepseek-v4-flash and qwen3.8-27b, which return the actions; the main session's own actions
are the key.

    python3 tools/lab_wake_trial.py extract [--since ISO --until ISO --n 40 --seed 0]   -> wakes.jsonl
    python3 tools/lab_wake_trial.py run [--models deepseek,qwen]                        -> wake_answers.jsonl
    python3 tools/lab_wake_trial.py report [--json]
    python3 tools/lab_wake_trial.py reviews-extract|reviews-run|reviews-report [--n 40 --cap 80000]

Design reviews (the second candidate): real design-reviewer runs of the window with a recorded verdict; each lab
model judges, from the reviewer's instructions (.claude/agents/design-reviewer.md) and either the design alone (the
brief and the first plan show|design output) or everything the Opus reviewer read (capped), and its PASS/FIX is
compared with the reviewer's; beside it, how many an always-PASS reviewer gets right.

A wake: the trigger text in full, the outputs of the state reads the main session made in that segment (plan.py
owners|status|show|next|unanswered, site events|items|decisions|runs|digest without --wait; what a dispatcher would
read itself), and the key: the agents it messaged (SendMessage to), launched (Agent: count), the events it acked
(site ack E<n>), the plan records it wrote (plan.py verbs) and whether it told Joaquin something (its visible text
over 60 characters, not "nothing new"). Correct = all five match; beside it, the share a dispatcher doing nothing
would get right. Our tokens of a wake: input + output + cache read + cache write of its messages (the unit
of tools/token_share.py).
"""
import argparse, glob, json, os, random, re, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_turns as mt  # noqa: E402

PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
# samples and answers live outside the checkout: another agent's `git stash -u` there unlinks untracked files under
# running writers (2026-11-25 01:06 UTC: the replay's answers went to deleted inodes)
DATA = os.path.expanduser("~/.local/share/fa_labllm/wake_trial")
os.makedirs(DATA, exist_ok=True)
URL = "https://lpnkarim.in2p3.fr/api/chat/completions"  # the /openai/v1 passthrough is disabled (403)
MODELS = {"deepseek": "deepseek-v4-flash", "qwen": "qwen3.8-27b"}
STATE = re.compile(r"tools/plan\.py (owners|status|show|next|unanswered|where)|\bsite (events|items|decisions|runs|"
                   r"digest)\b")
PLANW = re.compile(r"tools/plan\.py (run|wait|unwait|owner|evidence|finding|set-next|add-next|drop-next|answer|ask|"
                   r"withdraw|done|handoff|target|design-pass|design-fix|dead-end)\b")
ACK = re.compile(r"\bsite ack (E\d+)")
RULES = """You are the dispatcher of a research project's agent setup. A wake (an event of the job supervisor, an
agent finishing, or an agent's hand-back) arrives; decide the actions. Rules:
- An event goes to the agent that owns its work (an entry's owner: SendMessage to its agent id), or a mechanical
  worker when there is none; a READY goes to its owner; a hook incident to a worker; any other event is explained
  and acked (site ack E<n>) by whoever it reaches. Routine run ends never reach Joaquin.
- An owner's hand-back starting "ROUTINE:" never reaches Joaquin. An agent's findings and questions reach Joaquin
  rewritten once; a wake with nothing new for Joaquin ends with "Joaquin, nothing new."
- A stopped owner is resumed with SendMessage on every event of its work; a new agent (Agent) is launched only when
  the work has no live owner.
Answer with one JSON object only: {"send_to": [agent ids messaged], "launch": <number of new agents>,
"ack": ["E<n>", ...], "tell_joaquin": true|false, "plan": [verbs of `python3 tools/plan.py <verb>` to record, e.g.
"run", "wait", "owner", "answer"]}."""


def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [round(c - h, 3), round(c + h, 3)]


def text_of(c):
    return mt.text_of(c) if not isinstance(c, str) else c


def segments(since, until):
    """Main-session segments with full trigger text, tool calls (name, input), state outputs and tokens."""
    out, seen = [], set()
    for p in sorted(glob.glob(PROJ + "/*.jsonl")):
        cur, uses, bg = None, {}, {}
        for line in open(p, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t, typ = d.get("timestamp") or "", d.get("type")
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            if typ == "user":
                c = m.get("content")
                res = [b for b in c if isinstance(b, dict) and b.get("type") == "tool_result"] if isinstance(c, list) else []
                if res:
                    for b in res:
                        if cur is not None and b.get("tool_use_id") in uses:
                            cur["state"].append((uses[b["tool_use_id"]], text_of(b.get("content"))[:6000]))
                    continue
                txt = text_of(c)
                if txt.strip() and not d.get("isMeta"):
                    cur = {"file": os.path.basename(p), "t": t[:19], "trigger": mt.trigger_of(txt, d.get("origin"), bg),
                           "text": txt[:8000], "calls": [], "state": [], "said": "", "tokens": 0}
                    uses = {}
                    if since <= t[:16] < until:
                        out.append(cur)
                continue
            if typ != "assistant" or cur is None:
                continue
            u = m.get("usage") or {}
            if m.get("id") and u and m["id"] not in seen:
                seen.add(m["id"])
                cur["tokens"] += sum(u.get(k) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                              "cache_creation_input_tokens"))
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    cur["said"] += b.get("text", "")
                elif b.get("type") == "tool_use":
                    i = b.get("input") or {}
                    cmd = str(i.get("command") or "")
                    if b.get("name") == "Bash" and i.get("run_in_background"):
                        bg[b.get("id")] = cmd
                    cur["calls"].append((b.get("name"), i))
                    if b.get("name") == "Bash" and STATE.search(cmd) and "--wait" not in cmd:
                        uses[b.get("id")] = cmd[:300]
    return out


def key(s):
    said = s["said"].strip()
    return {"send": any(n == "SendMessage" for n, _ in s["calls"]),
            "send_ids": sorted({str(i.get("to")) for n, i in s["calls"] if n == "SendMessage"}),
            "launch": sum(n in ("Agent", "Task") for n, _ in s["calls"]),
            "ack": sorted({a for n, i in s["calls"] if n == "Bash" for a in ACK.findall(str(i.get("command") or ""))}),
            "tell_joaquin": len(said) > 60 and "nothing new" not in said.lower(),
            "plan": sorted({v for n, i in s["calls"] if n == "Bash" for v in PLANW.findall(str(i.get("command") or ""))})}


def extract(a):
    segs = [s for s in segments(a.since, a.until) if s["trigger"] in ("waiter", "agent_done", "handback")
            and mt.did_of([(n, str(i.get("command") or "")) for n, i in s["calls"]]) in ("orchestration", "none")]
    random.Random(a.seed).shuffle(segs)
    pick = segs[:a.n]
    with open(os.path.join(DATA, "wakes.jsonl"), "w") as f:
        for s in pick:
            f.write(json.dumps({"file": s["file"], "t": s["t"], "trigger": s["trigger"], "text": s["text"],
                                "state": s["state"], "tokens": s["tokens"], "key": key(s)}) + "\n")
    print("%d routing wakes in the window, %d sampled -> %s" % (len(segs), len(pick), os.path.join(DATA, "wakes.jsonl")))
    print("population tokens %.1fM, mean %.0fk" % (sum(s["tokens"] for s in segs) / 1e6,
                                                  sum(s["tokens"] for s in segs) / max(len(segs), 1) / 1e3))


def ask(model, prompt, system=RULES):
    k = open(os.path.expanduser("~/.config/lpnhe/llm_key")).read().strip()
    r = urllib.request.Request(URL, data=json.dumps({"model": model, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + k, "Content-Type": "application/json", "User-Agent": "generic-client/1.0"})
    for i in range(5):  # a 502/503 from the proxy comes back at once: retried, so one busy spell does not skip a pass
        t = time.time()
        try:
            d = json.load(urllib.request.urlopen(r, timeout=1200))
            break
        except urllib.error.HTTPError as e:
            if e.code not in (502, 503) or i == 4:
                raise
            time.sleep(120)
    return d["choices"][0]["message"].get("content") or "", d.get("usage") or {}, time.time() - t


def parse(ans):
    m = re.search(r"\{.*\}", ans, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except ValueError:
        j = {}
    return {"send": bool(j.get("send_to")), "send_ids": sorted({str(x) for x in j.get("send_to") or []}),
            "launch": int(j.get("launch") or 0),
            "ack": sorted({str(x) for x in j.get("ack") or []}), "tell_joaquin": bool(j.get("tell_joaquin")),
            "plan": sorted({str(x) for x in j.get("plan") or []})}


def answered(path, key):
    """Instances already answered (no ERROR) in an answers file: a rerun resumes."""
    if not os.path.exists(path):
        return set()
    return {key(r) for r in (json.loads(l) for l in open(path)) if not r["answer"].startswith("ERROR")}


def run(a):
    wakes = [json.loads(l) for l in open(os.path.join(DATA, "wakes.jsonl"))]
    done = answered(os.path.join(DATA, "wake_answers.jsonl"), lambda r: (r["t"], r["model"]))
    with open(os.path.join(DATA, "wake_answers.jsonl"), "a") as f:
        for w in wakes:
            prompt = "WAKE (%s, %s):\n%s\n\nSTATE READ:\n%s" % (w["trigger"], w["t"], w["text"], "\n\n".join(
                "$ %s\n%s" % (c, o) for c, o in w["state"]) or "(none)")
            for name in a.models.split(","):
                if (w["t"], name) in done:
                    continue
                try:
                    ans, u, sec = ask(MODELS[name], prompt)
                except Exception as e:  # a lab outage: recorded, not scored
                    ans, u, sec = "ERROR %s" % e, {}, 0
                f.write(json.dumps({"t": w["t"], "model": name, "answer": ans[-4000:], "got": parse(ans),
                                    "usage": u, "seconds": round(sec, 1)}) + "\n")
                f.flush()


def over(a, b):
    """Whether field value a holds something b lacks: a set difference for id lists, a > b for flags and counts."""
    return bool(set(a) - set(b)) if isinstance(a, list) else a > b


def report(a):
    wakes = {w["t"]: w for w in (json.loads(l) for l in open(os.path.join(DATA, "wakes.jsonl")))}
    rows = [json.loads(l) for l in open(os.path.join(DATA, "wake_answers.jsonl"))]
    last = {(r["t"], r["model"]): r for r in rows if r["t"] in wakes}
    rep = {}
    for name in sorted({m for _, m in last}):
        rs = [r for (t, m), r in last.items() if m == name and not r["answer"].startswith("ERROR")]
        fields = ("send", "launch", "ack", "tell_joaquin", "plan")
        ok = [r for r in rs if all(r["got"][k] == wakes[r["t"]]["key"][k] for k in fields)]
        quiet = [r for r in rs if not any(wakes[r["t"]]["key"][k] for k in fields)]
        ids = [r for r in rs if wakes[r["t"]]["key"]["send_ids"] and all(
            i in wakes[r["t"]]["text"] + json.dumps(wakes[r["t"]]["state"]) for i in wakes[r["t"]]["key"]["send_ids"])]
        rep[name] = {"n": len(rs), "correct": len(ok), "correct_wilson": wilson(len(ok), len(rs)),
                     "do_nothing_baseline": len(quiet), "do_nothing_wilson": wilson(len(quiet), len(rs)),
                     "wrong_on_quiet": sum(r not in ok for r in quiet),
                     "active": len(rs) - len(quiet), "correct_on_active": sum(r in ok for r in rs if r not in quiet),
                     "exact_ids_when_in_prompt": [sum(r["got"]["send_ids"] == wakes[r["t"]]["key"]["send_ids"]
                                                      for r in ids), len(ids)],
                     "per_field": {k: sum(r["got"][k] == wakes[r["t"]]["key"][k] for r in rs) for k in fields},
                     # sensitivity: the prompt's example plan verbs prime answers, the key has none on 95/100
                     "correct_no_plan": sum(all(r["got"][k] == wakes[r["t"]]["key"][k] for k in fields[:4]) for r in rs),
                     "do_nothing_no_plan": sum(not any(wakes[r["t"]]["key"][k] for k in fields[:4]) for r in rs),
                     "missed": {k: sum(over(wakes[r["t"]]["key"][k], r["got"][k]) for r in rs)
                                for k in fields[:4]},
                     "extra": {k: sum(over(r["got"][k], wakes[r["t"]]["key"][k]) for r in rs)
                               for k in fields[:4]},
                     "wrong": [(r["t"], wakes[r["t"]]["key"], r["got"]) for r in rs if r not in ok],
                     "seconds_median": sorted(r["seconds"] for r in rs)[len(rs) // 2] if rs else None,
                     "our_tokens_of_these_wakes": sum(wakes[r["t"]]["tokens"] for r in rs)}
    print(json.dumps(rep, indent=1) if a.json else "\n".join("%s: %d/%d correct %s, %.0f s median" % (
        k, v["correct"], v["n"], v["per_field"], v["seconds_median"] or 0) for k, v in rep.items()))
    json.dump(rep, open(os.path.join(DATA, "wake_summary.json"), "w"), indent=1)
    print(json.dumps({k: {f: x for f, x in v.items() if f != "wrong"} for k, v in rep.items()}))  # one line for % claim:


REVIEWER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".claude", "agents",
                        "design-reviewer.md")
VERDICT = re.compile(r"tools/plan\.py design-(pass|fix)\b")


def reviews_extract(a):
    """Design review rounds of the window: a reviewer agent is resumed for each re-check, so the unit is one round
    (from a prompt to the next): its prompt, the first plan show|design output after it (the design), only that
    round's tool outputs (the material), and that round's own design-pass|design-fix verdict. Rounds with no fresh
    design read or no verdict are dropped."""
    rows, nall, nverdict = [], 0, 0
    for mp in glob.glob(PROJ + "/*/subagents/agent-*.meta.json"):
        if json.load(open(mp)).get("agentType") != "design-reviewer":
            continue
        rounds, cur, uses = [], None, {}
        for line in open(mp[:-10] + ".jsonl", errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            c = m.get("content")
            if d.get("type") == "user":
                res = [b for b in c if isinstance(b, dict) and b.get("type") == "tool_result"] if isinstance(c, list) else []
                if not res:
                    txt = text_of(c)
                    if txt.strip() and not txt.lstrip().startswith("<system-reminder>"):
                        cur = {"t": (d.get("timestamp") or "")[:19], "brief": txt, "design": None, "material": [],
                               "verdict": None}
                        rounds.append(cur)
                        uses = {}
                    continue
                for b in res:
                    # material stops at the round's verdict call: later reads can print the round's own verdict
                    if cur is not None and cur["verdict"] is None and b.get("tool_use_id") in uses:
                        txt = text_of(b.get("content"))
                        cur["material"].append((uses[b["tool_use_id"]], txt))
                        # the design: the round's first read holding a design's measure and outcome (a saved-output
                        # placeholder or a slice of the entry without them is not one)
                        if cur["design"] is None and re.search(r"\bmeasure:", txt) and re.search(r"\boutcome:", txt):
                            cur["design"], cur["design_cmd"] = txt, uses[b["tool_use_id"]]
            elif d.get("type") == "assistant" and cur is not None:
                for b in c or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        cmd = str((b.get("input") or {}).get("command") or (b.get("input") or {}).get("file_path") or "")
                        v = VERDICT.search(cmd)
                        if v:
                            cur["verdict"] = v.group(1).upper()
                        else:
                            uses[b.get("id")] = cmd[:300]
        for k, r in enumerate(rounds):
            if a.since <= r["t"][:16] < a.until:
                nall += 1
                nverdict += bool(r["verdict"])
            if r["design"] and r["verdict"] and a.since <= r["t"][:16] < a.until:
                rows.append({"agent": "%s#%d" % (os.path.basename(mp)[6:-16], k), "t": r["t"], "brief": r["brief"][:6000],
                             "design": r["design"][:30000], "design_cmd": r.get("design_cmd", ""),
                             "slice": bool(re.search(r"\b(sed|grep|awk)\b", r.get("design_cmd", ""))), "material": r["material"], "verdict": r["verdict"]})
    random.Random(a.seed).shuffle(rows)
    with open(os.path.join(DATA, "reviews.jsonl"), "w") as f:
        for r in rows[:a.n]:
            f.write(json.dumps(r) + "\n")
    print("%d design review rounds in the window, %d with a verdict, %d of them with a read holding the design's measure "
          "and outcome (%d FIX), %d kept" % (nall, nverdict, len(rows), sum(r["verdict"] == "FIX" for r in rows),
                                              min(a.n, len(rows))))


def reviews_run(a):
    rules = open(REVIEWER).read()
    rules = rules.split("---", 2)[-1] + "\n\nYou cannot run commands: judge from what is given. End with one line " \
        "'VERDICT: PASS' or 'VERDICT: FIX', after the reasons."
    done = answered(os.path.join(DATA, "review_answers.jsonl"), lambda r: (r["agent"], r["arm"], r["model"]))
    k, n = (int(x) for x in a.shard.split("/"))
    with open(os.path.join(DATA, "review_answers.jsonl"), "a") as f:
        for i, r in enumerate(json.loads(l) for l in open(os.path.join(DATA, "reviews.jsonl"))):
            if i % n != k:  # --shard k/n: every n-th round from the k-th, for parallel streams
                continue
            mat = ""
            for cmd, out in r["material"]:
                if len(mat) > a.cap:
                    break
                mat += "$ %s\n%s\n\n" % (cmd, out[:a.cap - len(mat)])
            arms = {"design only": "BRIEF:\n%s\n\nDESIGN:\n%s" % (r["brief"], r["design"]),
                    "reviewer's material": "BRIEF:\n%s\n\nMATERIAL READ:\n%s" % (r["brief"], mat)}
            for arm, prompt in arms.items():
                for name in a.models.split(","):
                    if (r["agent"], arm, name) in done:
                        continue
                    try:
                        ans, u, sec = ask(MODELS[name], prompt, rules)
                    except Exception as e:
                        ans, u, sec = "ERROR %s" % e, {}, 0
                    v = re.findall(r"VERDICT:\s*\**\s*(PASS|FIX)", ans)
                    f.write(json.dumps({"agent": r["agent"], "arm": arm, "model": name, "verdict": v[-1] if v else None,
                                        "answer": ans[-6000:], "usage": u, "seconds": round(sec, 1)}) + "\n")
                    f.flush()


def reviews_report(a):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "reviews.jsonl"))]
    key = {r["agent"]: r["verdict"] for r in rows}
    sl = {r["agent"]: r.get("slice", False) for r in rows}
    last = {}
    for r in (json.loads(l) for l in open(os.path.join(DATA, "review_answers.jsonl"))):
        if r["agent"] in key and not r["answer"].startswith("ERROR"):
            last[(r["agent"], r["arm"], r["model"])] = r
    npass = sum(v == "PASS" for v in key.values())
    rep = {"n": len(key), "fix_in_key": len(key) - npass, "always_pass_correct": npass,
           "always_pass_wilson": wilson(npass, len(key))}
    for arm in sorted({k[1] for k in last}):
        for m in sorted({k[2] for k in last}):
            rs = [r for (g, aa, mm), r in last.items() if aa == arm and mm == m]
            rep["%s / %s" % (m, arm)] = {
                "n": len(rs), "agree": sum(r["verdict"] == key[r["agent"]] for r in rs),
                "agree_wilson": wilson(sum(r["verdict"] == key[r["agent"]] for r in rs), len(rs)),
                "fix_missed": sum(r["verdict"] != "FIX" and key[r["agent"]] == "FIX" for r in rs),
                "fix_caught": sum(r["verdict"] == "FIX" and key[r["agent"]] == "FIX" for r in rs),
                "false_fix": sum(r["verdict"] == "FIX" and key[r["agent"]] == "PASS" for r in rs),
                "no_verdict": sum(r["verdict"] is None for r in rs),
                "seconds_median": sorted(r["seconds"] for r in rs)[len(rs) // 2] if rs else None,
                # the design-only arm sees a slice of the entry when the reviewer's design read was sed/grep/head
                "agree_by_design_read": {w: [sum(r["verdict"] == key[r["agent"]] for r in rs if sl[r["agent"]] == (w == "slice")),
                                             sum(sl[r["agent"]] == (w == "slice") for r in rs)] for w in ("full", "slice")}}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(DATA, "review_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--since", default="2026-11-09T00:00")
    p.add_argument("--until", default="2026-11-23T00:00")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("run")
    p.add_argument("--models", default="deepseek,qwen")
    p = sp.add_parser("report")
    p.add_argument("--json", action="store_true")
    for n in ("reviews-extract", "reviews-run", "reviews-report"):
        p = sp.add_parser(n)
        p.add_argument("--since", default="2026-11-09T00:00")
        p.add_argument("--until", default="2026-11-23T00:00")
        p.add_argument("--n", type=int, default=40)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--models", default="deepseek,qwen")
        p.add_argument("--cap", type=int, default=80000, help="characters of the reviewer's material handed over")
        p.add_argument("--shard", default="0/1", help="k/n: this stream takes every n-th round from the k-th")
    a = ap.parse_args()
    {"extract": extract, "run": run, "report": report, "reviews-extract": reviews_extract, "reviews-run": reviews_run,
     "reviews-report": reviews_report}[a.cmd](a)


if __name__ == "__main__":
    main()
