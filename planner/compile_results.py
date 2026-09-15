"""Compile all experiment outputs from this repo into a single RESULTS.txt:
  1. gpt-5.4-nano tabletop baseline (20 held-out commands) + proxy metrics + token/cost
  2. Table I & II reproduction (dataset complexity) vs the paper
  3. Fig. 8 / Fig. 9 sample reproductions
"""
import json
import re
from pathlib import Path

from reproduce_tables import load_text_sets, table1_row, pos_entropy, PAPER_T1, PAPER_T2

ROOT = Path(__file__).resolve().parent.parent  # repo root (planner/ -> ..)
BASELINE = ROOT / "baseline_tabletop_gpt-5.4-nano.json"
FIG89 = ROOT / "fig89_samples_gpt-5.4-nano.txt"
OUT = ROOT / "RESULTS.txt"

PRICE_IN, PRICE_OUT = 0.20, 1.25  # gpt-5.4-nano $ per 1M tokens
STEP_RE = re.compile(r"\(ACTION:\s*([A-Za-z]+)\s*\|\s*TARGET:", re.I)


def section_baseline(w):
    data = json.loads(BASELINE.read_text())
    # New run-document shape is a dict with "results"; tolerate a bare list too.
    results = data["results"] if isinstance(data, dict) else data
    model = data.get("model", "gpt-5.4-nano") if isinstance(data, dict) else "gpt-5.4-nano"
    w("=" * 72 + f"\n1. BASELINE — {model}, tabletop, Fig.5 prompt, {len(results)} held-out commands\n" + "=" * 72 + "\n\n")
    fmt_ok = only_pp = alt_ok = 0
    tin = tout = 0
    for x in results:
        acts = [m.group(1).upper() for m in STEP_RE.finditer(x["model_output"])]
        if acts:
            fmt_ok += 1
        if acts and all(a in ("PICK", "PLACE") for a in acts):
            only_pp += 1
        ok = bool(acts) and acts[0] == "PICK"
        for a, b in zip(acts, acts[1:]):
            if a == "PICK" and b != "PLACE":
                ok = False
            if a == "PLACE" and b != "PICK":
                ok = False
        if ok:
            alt_ok += 1
        u = x.get("usage", {})
        tin += u.get("input", u.get("prompt_tokens", 0))
        tout += u.get("output", u.get("completion_tokens", 0))
        gt = x["ground_truth_steps"]
        gt = "\n".join(gt) if isinstance(gt, list) else gt
        w(f"[{x.get('index','?')}] COMMAND: {x['command']}\n")
        w(f"    OBJECTS: {x['objects_on_table']}\n")
        w(f"    --- {model} output ---\n")
        w("    " + x["model_output"].replace("\n", "\n    ") + "\n")
        w("    --- ground-truth steps ---\n")
        w("    " + gt.replace("\n", "\n    ") + "\n\n")

    n = len(results)
    cost = tin * PRICE_IN / 1e6 + tout * PRICE_OUT / 1e6
    w("-" * 72 + "\nPROXY METRICS (structural validity; NOT the paper's CLIPort success rate)\n" + "-" * 72 + "\n")
    w(f"  parseable ACTION/TARGET format : {fmt_ok}/{n}\n")
    w(f"  only PICK/PLACE (tabletop rule): {only_pp}/{n}\n")
    w(f"  valid PICK->PLACE alternation  : {alt_ok}/{n}\n\n")
    w("TOKEN USAGE & COST (gpt-5.4-nano @ $0.20/$1.25 per 1M in/out)\n")
    w(f"  input tokens : {tin}\n  output tokens: {tout}\n  total        : {tin + tout}\n")
    w(f"  est. cost    : ${cost:.4f} for {n} commands\n\n")


def section_tables(w):
    sets = load_text_sets()
    w("=" * 72 + "\n2. TABLE I & II REPRODUCTION (dataset complexity) — ours vs paper\n" + "=" * 72 + "\n\n")
    w("TABLE I\n")
    w(f"{'Dataset':22} | {'Kincaid':>16} | {'SynTree(var)':>22} | {'#Types':>14}\n")
    for name, sents in sets.items():
        k, m, v, ty = table1_row(name, sents)
        pk, pm, pv, pty = PAPER_T1[name]
        w(f"{name:22} | {k:6.2f} (p {pk:4.2f}) | {m:4.2f}({v:.2f}) (p {pm:.2f}({pv:.2f})) | {ty:5} (p {pty})\n")
    w("\nTABLE II  (POS entropy on commands, natural log / nats)\n")
    w(f"{'Dataset':22} | {'NN':>15} | {'VB':>15} | {'AD':>15}\n")
    for name in ["Tabletop, Command", "Kitchen, Command"]:
        e = pos_entropy(sets[name])
        pn, pv, pa = PAPER_T2[name]
        w(f"{name:22} | {e['NN']:5.2f} (p {pn:4.2f}) | {e['VB']:5.2f} (p {pv:4.2f}) | {e['AD']:5.2f} (p {pa:4.2f})\n")
    w("\nNote: spaCy metrics on a 4,000-sentence sample; Kincaid/#Types on full corpus.\n\n")


def section_fig89(w):
    w("=" * 72 + "\n3. FIG. 8 / FIG. 9 SAMPLE REPRODUCTIONS (gpt-5.4-nano)\n" + "=" * 72 + "\n\n")
    w(FIG89.read_text() if FIG89.exists() else "(fig89 file missing)\n")


def main():
    parts = []
    w = parts.append
    w("CONSOLIDATED RESULTS — small-LMs task-planning reproduction\n")
    w("Paper: Choi & Ahn, arXiv 2404.03891.  Baseline model: gpt-5.4-nano.\n\n")
    section_baseline(w)
    section_tables(w)
    section_fig89(w)
    OUT.write_text("".join(parts))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
