#!/usr/bin/env python3
"""Lookup trial (agent_redesign steps 26 and 29): ten past code lookups of FA (where is X computed, who calls Y),
each answered by a low-effort Explore agent per arm, the arms differing only in the tool the brief tells it to use.

    python3 tools/lookup_trial.py brief TASK ARM        the brief given to the agent (question identical across arms)
    python3 tools/lookup_trial.py record TASK ARM AGENT_ID [--hand ok|wrong]   score the agent's answer, count its tokens and tool calls
    python3 tools/lookup_trial.py report [--arms A,B] [--fig BASE]   per arm: correct, tokens and tool calls per task; the figure

Arms: grep (as now), lsp (pyright LSP tool), graph (graphify CLI queries via Bash), batch (tools/batch.py).
Tokens of an agent = sum over its API calls (deduplicated by message id) of input + cache read + cache write + output,
as tools/wake_batching.py counts them. Correct (code) = each expected (file, function) named on
one answer line, files matched by whole path components and functions exactly, and for a who-calls task no file
outside the expected set; tool calls = distinct tool_use ids on assistant lines. Recall answers: a blind hand read
decides (record --hand ok|wrong); the substring check is an aid. Records: analysis/agent_redesign/lookup_trial.json.
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "analysis", "agent_redesign", "lookup_trial.json")
SUB = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA/*/subagents/agent-%s.jsonl")
ARMS = ("grep", "grep2", "lsp", "graph", "batch", "now", "now2", "recall")   # grep2, now2: control repeats

# step 27: past cases where Joaquin's earlier words were lost; id: (question, substrings the answer must hold, None)
RECALL = {
    "context_500k": ("On 2026-11-04 Joaquin wrote 'The decision to increase the context to 500k was made for a reason'. "
                     "Find his own earlier words that made that decision, verbatim, with their date and time.",
                     ["500k for you", "300k for the others", "2026-10-31"], None),
    "batch_10min": ("On 2026-11-09 Joaquin wrote 'I though the 10 minute batching was implemented'. Find his own "
                    "earlier words that agreed to the 10-minute batching, verbatim, with their date and time.",
                    ["10 minutes seems reasonable", "2026-11-07"], None),
    "no_agent": ("On 2026-11-11 at 03:24 UTC Joaquin wrote 'I didn't say \"no agents\", I said \"no agent\" to the "
                 "agent recalculating the gpu hours'. Find the message of his this refers to, verbatim, with its "
                 "date and time.", ["no agent, stop that", "03:21"], None),
    "wording_yes": ("On 2026-11-18 the main session told Joaquin that an owner changed two results.tex paragraphs "
                    "(Muon's lead no longer 'grows with the horizon'; the particle-ID token '14-20% lower') and he "
                    "replied 'I don't remember saying yes to this'. Did he ever say yes to that change? Name the "
                    "record that asked him (entry and date) and quote any words of his on it.",
                    ["zng_uq_benchmark", "2026-11-17", "never"], None),
    "b1_n1_scope": ("For the Z+ng protocol test of 2026-11-19, what did Joaquin approve for arm (a)'s "
                    "hyperparameters? Quote his words and the decision's recorded scope, with date and time, and say "
                    "whether the design that ran used the same hyperparameters.",
                    ["those runs can go", "n=1 searched", "each n"], None),
}

# id: (question, expected "file:function" strings, who-calls: the allowed file set for precision, else None)
_CACHED = ["mix_compose.py", "datagen.py", "prebuild_recipes.py", "mg5_pipeline_final.py", "tools/lhe_parts.py",
           "tools/prune_mixture_pools.py", "tools/pool_physics.py", "tools/check_site_data.py", "sweep/sweep_manager.py"]
TASKS = {
    "assemble_callers": ("Which non-test code calls mg5_pipeline_final.lhe_assemble? Give each caller's file and "
                         "enclosing function.", ["tools/lhe_parts.py:_run_task"],
                         ["tools/lhe_parts.py"]),
    "is_cached_callers": ("List every non-test Python file that calls datagen._is_cached (datagen.py itself "
                          "included if it calls it).", _CACHED, _CACHED),
    "val_combine": ("Where, during validation, is the combined val_loss_no_reg computed from the per-process "
                    "losses? Give the file and method.", ["experiment.py:_validate"], None),
    "ckpt_select": ("Where in the training loop is the best checkpoint chosen on the lowest val_loss_no_reg? "
                    "Give the file and method.", ["base_experiment.py:train"], None),
    "het_loss": ("Where is the heteroscedastic (Gaussian NLL with predicted sigma) loss function defined? Give the "
                 "file and function.", ["losses.py:heteroscedastic_loss"], None),
    "fold_spool": ("In the Claude hooks of this repo, which function decides whether an owner's hand-back arriving "
                   "while the main session is mid-turn is folded into the turn instead of spooled? Give the file "
                   "and function.", [".claude/hooks/notification_absorb.py:mid_turn"], None),
    "freeze_callers": ("Which non-test code calls _freeze_all_but_sigma_head? Give each caller's file and "
                       "enclosing method.", ["experiment.py:init_model"], ["experiment.py"]),
    "f_default": ("Where is a recipe's sampling.f_default share (the default-source part of a mixture pool) applied "
                  "when the pool is composed at load time? Give the file and function.",
                  ["mix_compose.py:compose"], None),
    "lhe_order": ("Where is the order of an LHE pool's val, test and train blocks laid out? Give the file and "
                  "function.", ["mg5_pipeline_final.py:lhe_layout"], None),
    "sigma_cfg_callers": ("Which non-test code calls run.sigma_stage_cfg? Give each caller's file and enclosing "
                          "function.", ["run.py:main", "tools/sigma_stage.py"], ["run.py", "tools/sigma_stage.py"]),
}
TASKS.update(RECALL)

TOOL = {
    "grep": "Use Grep, Glob and Read (or grep via Bash) as you normally would.",
    "lsp": ("Use the LSP tool (pyright) for definitions, references and call hierarchy: load it first with ToolSearch "
            "query 'select:LSP'. Fall back to grep only for what LSP cannot answer."),
    "graph": ("Use the graphify CLI via Bash on the prebuilt code graph: "
              "cd /home/joaquin/work/FA/worktrees/wt-graphify && .venv-graphify/bin/graphify --help shows the query "
              "commands (graph in ../../../graphify-out-FA). Fall back to grep only for what the graph cannot answer."),
    "batch": ("Batch your read-only lookups: python3 /home/joaquin/work/FA/worktrees/wt-batch/tools/batch.py takes "
              "several operations in one Bash call (run it with --help first). Prefer one batch call over several "
              "single calls."),
    "now": ("Search as you normally would: Joaquin's messages are in the main session transcript "
            "~/.claude/projects/-home-joaquin-work-FA/71602535-0955-42ad-a58f-23dc320798dc.jsonl (JSON lines; his own "
            "messages are type user with turnOrigin human), the plan via python3 tools/plan.py show <entry>, the ledger via "
            "site decisions --project FA --all."),
    "recall": ("Use python3 /home/joaquin/work/FA/worktrees/wt-recall/tools/recall.py (run --help first): search "
               "finds summary records, expand shows the raw transcript lines behind one. Fall back to other searches "
               "only for what it cannot answer."),
}

TOOL["grep2"], TOOL["now2"] = TOOL["grep"], TOOL["now"]


def brief(task, arm):
    q = TASKS[task][0]
    if task in RECALL:
        return ("Answer one question about this project's history from its records (read-only; do not change any "
                "file, plan entry or decision). " + TOOL[arm] + "\nQuestion: " + q +
                "\nReply with only the answer: the quoted words, their date and time, and one line of context.")
    return ("Answer one question about the code in /home/joaquin/work/FA (ignore worktrees/, tests/, .venv*). "
            + TOOL[arm] + "\nQuestion: " + q +
            "\nReply with only the answer: one line per item, as path/to/file.py:function_name, no explanation.")


def usage(agent):
    """(tokens, tool calls, final answer text) of one agent's transcript."""
    seen, seen_tools, tok, calls, last = set(), set(), 0, 0, ""
    for f in glob.glob(SUB % agent):
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message")
            if d.get("type") != "assistant" or not isinstance(m, dict):
                continue   # api-request-blob lines repeat each call
            for c in m.get("content") or []:
                if c.get("type") == "tool_use" and c.get("id") not in seen_tools:
                    seen_tools.add(c.get("id"))
                    calls += 1
                    if c.get("name") == "SubagentHandback":
                        calls -= 1
                        last = c["input"].get("message", last)
                elif c.get("type") == "text" and c.get("text", "").strip():
                    last = c["text"]
            if m.get("usage") and m.get("id") and m["id"] not in seen:
                seen.add(m["id"])
                u = m["usage"]
                tok += (u.get("input_tokens", 0) + u.get("output_tokens", 0) + u.get("cache_read_input_tokens", 0)
                        + u.get("cache_creation_input_tokens", 0))
    return tok, calls, last


def _pairs(answer):
    """(path, function) per answer line: path relative to the checkout, function the last dotted name or ''."""
    out = []
    for line in answer.replace("`", "").splitlines():
        m = re.search(r"((?:file://)?[\w./-]+\.py)(?::(?:\d+:?)?\s*([\w.<>]+))?", line)
        if m:
            path = re.sub(r"^(?:file://)?/home/joaquin/work/FA/", "", m.group(1))
            path = re.sub(r"^worktrees/[^/]+/", "", path)
            path = path[2:] if path.startswith("./") else path
            out.append((path, (m.group(2) or "").split(".")[-1]))
    return out


def _same(path, want):
    return path == want   # relative to the checkout (_pairs strips the checkout and worktree prefixes)


def correct(task, answer):
    _, want, allowed = TASKS[task]
    if task in RECALL:   # aid only: a blind hand read of every recall answer decides (record --hand)
        low = answer.lower().replace("\u2019", "'")
        return all(w.lower() in low for w in want)
    pairs = _pairs(answer)
    for w in want:
        f, _, fn = w.partition(":")
        if not any(_same(p, f) and (not fn or q == fn) for p, q in pairs):
            return False
    if allowed is not None:   # who-calls: no caller outside the expected files (tests excluded by the question)
        return all(any(_same(p, x) for x in allowed) for p, _ in pairs)
    return True


def load():
    return json.load(open(REC)) if os.path.exists(REC) else {}


def record(task, arm, agent, hand=None):
    tok, calls, ans = usage(agent)
    r = load()
    ok = correct(task, ans) if hand is None else hand == "ok"
    r.setdefault(task, {})[arm] = {"agent": agent, "tokens": tok, "tool_calls": calls, "answer": ans,
                                   "correct": ok, "auto": correct(task, ans), "hand": hand}
    os.makedirs(os.path.dirname(REC), exist_ok=True)
    json.dump(r, open(REC, "w"), indent=1)
    print(task, arm, r[task][arm]["correct"], tok, calls)


def report(fig=None, arms=None):
    r = load()
    arms = arms or [a for a in ARMS if any(a in v for v in r.values())]
    tasks = [t for t in TASKS if all(a in r.get(t, {}) for a in arms)]   # paired tasks only
    out = {}
    for a in arms:
        rows = [r[t][a] for t in tasks]
        out[a] = {"tasks": len(rows), "correct": sum(x["correct"] for x in rows),
                  "tokens_total": sum(x["tokens"] for x in rows), "tool_calls_total": sum(x["tool_calls"] for x in rows)}
    print(json.dumps({"paired_tasks": tasks, "arms": out}))
    if fig:
        sys.path.insert(0, ROOT)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import plot_style  # noqa: F401
        except ImportError:
            pass
        fig_, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        x = range(len(tasks))
        w = 0.8 / len(arms)
        for i, a in enumerate(arms):
            xs = [k + (i - (len(arms) - 1) / 2) * w for k in x]
            for ax, key, sc in ((axs[0], "tokens", 1e-3), (axs[1], "tool_calls", 1)):
                ax.bar(xs, [r[t][a][key] * sc for t in tasks], w, label=a)
                bad = [(xx, r[t][a][key] * sc) for xx, t in zip(xs, tasks) if not r[t][a]["correct"]]
                if bad:
                    ax.plot(*zip(*bad), "kx", ms=6, label="wrong" if i == 0 else None)
        axs[0].set_yscale("log")
        axs[0].set_ylabel("tokens to answer [k]")
        axs[1].set_ylabel("tool calls to answer")
        for ax in axs:
            ax.set_xticks(list(x))
            ax.set_xticklabels(tasks, rotation=60, ha="right", fontsize=7)
        axs[1].legend(fontsize=7)
        fig_.tight_layout()
        for ext in ("png", "pdf"):
            fig_.savefig(fig + "." + ext, dpi=150)
        h, l = axs[1].get_legend_handles_labels()
        open(fig + ".legend.txt", "w").write("\n".join(l) + "\n")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "brief":
        print(brief(sys.argv[2], sys.argv[3]))
    elif cmd == "record":
        a = sys.argv
        record(*a[2:5], hand=a[a.index("--hand") + 1] if "--hand" in a else None)
    elif cmd == "report":
        a = sys.argv
        report(a[a.index("--fig") + 1] if "--fig" in a else None,
               a[a.index("--arms") + 1].split(",") if "--arms" in a else None)
