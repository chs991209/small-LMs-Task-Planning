"""Reproduce the paper's dataset-analysis results (Tables I & II) from the COST datasets.

Table I : Flesch-Kincaid grade, avg syntactic-tree depth (variance), vocabulary #types.
Table II: entropy per part-of-speech family (NN=NOUN/PROPN, VB=VERB, AD=ADJ/ADV) on commands.

spaCy-based metrics (tree depth, POS entropy) are computed on a fixed sample for speed;
Kincaid and #types use the full corpus. Numbers are approximate: tokenizer, spaCy model
version, and log base differ from the authors' setup.
"""
import json
import math
import re
import random
from pathlib import Path

import spacy
import textstat

ROOT = Path(__file__).resolve().parent.parent  # repo root (planner/ -> ..)
random.seed(42)
SAMPLE = 4000  # sentences for spaCy parse/POS metrics

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

STEP_PREFIX = re.compile(r"^\s*Step\s*\d+\.\s*", re.I)
ACTION_TAG = re.compile(r"\(ACTION:.*?\)\s*$", re.I | re.S)


def clean_step(s: str) -> str:
    s = STEP_PREFIX.sub("", s)
    s = ACTION_TAG.sub("", s)
    return s.strip()


def load_text_sets():
    tab = json.loads((ROOT / "datasets/dataset_tabletop.json").read_text())
    kit = json.loads((ROOT / "datasets/dataset_kitchen.json").read_text())

    def steps_sentences(d):
        out = []
        for item in d["steps"]:
            steps = item if isinstance(item, list) else [item]
            for s in steps:
                c = clean_step(s)
                if c:
                    out.append(c)
        return out

    return {
        "Tabletop, Command": tab["high_instructions"],
        "Tabletop, Steps": steps_sentences(tab),
        "Kitchen, Command": kit["high_instructions"],
        "Kitchen, Steps": steps_sentences(kit),
    }


def tree_depth(token) -> int:
    # depth of the subtree rooted at `token` (root token -> full sentence depth)
    if not list(token.children):
        return 1
    return 1 + max(tree_depth(c) for c in token.children)


def sentence_depth(doc) -> int:
    roots = [t for t in doc if t.head == t]
    return max((tree_depth(r) for r in roots), default=1)


def n_types(sentences) -> int:
    vocab = set()
    for doc in nlp.tokenizer.pipe(sentences, batch_size=1000):
        for t in doc:
            if t.is_alpha:
                vocab.add(t.text.lower())
    return len(vocab)


def table1_row(name, sentences):
    # ensure each item ends with terminal punctuation so textstat counts sentences correctly
    corpus = "\n".join(s.rstrip(" .!?") + "." for s in sentences)
    kincaid = textstat.flesch_kincaid_grade(corpus)

    sample = sentences if len(sentences) <= SAMPLE else random.sample(sentences, SAMPLE)
    depths = [sentence_depth(doc) for doc in nlp.pipe(sample, batch_size=500)]
    mean = sum(depths) / len(depths)
    var = sum((d - mean) ** 2 for d in depths) / len(depths)

    types = n_types(sentences)
    return kincaid, mean, var, types


def pos_entropy(sentences):
    # families: NN=noun, VB=verb, AD=adjective+adverb
    families = {"NN": ["NOUN", "PROPN"], "VB": ["VERB"], "AD": ["ADJ", "ADV"]}
    counts = {k: {} for k in families}
    sample = sentences if len(sentences) <= SAMPLE else random.sample(sentences, SAMPLE)
    for doc in nlp.pipe(sample, batch_size=500):
        for t in doc:
            for fam, tags in families.items():
                if t.pos_ in tags and t.is_alpha:
                    w = t.text.lower()
                    counts[fam][w] = counts[fam].get(w, 0) + 1

    out = {}
    for fam, c in counts.items():
        total = sum(c.values())
        h = -sum((n / total) * math.log(n / total) for n in c.values()) if total else 0.0
        out[fam] = h
    return out


PAPER_T1 = {
    "Tabletop, Command": (4.81, 5.71, 1.94, 1410),
    "Tabletop, Steps": (1.79, 4.45, 2.09, 990),
    "Kitchen, Command": (3.07, 4.60, 0.80, 2239),
    "Kitchen, Steps": (1.32, 3.84, 0.71, 2979),
}
PAPER_T2 = {
    "Tabletop, Command": (3.96, 3.29, 2.49),
    "Kitchen, Command": (5.34, 4.79, 3.54),
}


def main():
    sets = load_text_sets()

    print("\n=== TABLE I (ours vs paper) ===")
    print(f"{'Dataset':22} | {'Kincaid':>14} | {'SynTree(var)':>20} | {'#Types':>14}")
    print("-" * 80)
    for name, sents in sets.items():
        k, m, v, ty = table1_row(name, sents)
        pk, pm, pv, pty = PAPER_T1[name]
        print(f"{name:22} | {k:6.2f} (p {pk:4.2f}) | {m:4.2f}({v:.2f}) (p {pm:.2f}({pv:.2f})) | {ty:5} (p {pty})")

    print("\n=== TABLE II — POS entropy on commands (ours vs paper, nats) ===")
    print(f"{'Dataset':22} | {'NN':>13} | {'VB':>13} | {'AD':>13}")
    print("-" * 72)
    for name in ["Tabletop, Command", "Kitchen, Command"]:
        e = pos_entropy(sets[name])
        pn, pv, pa = PAPER_T2[name]
        print(f"{name:22} | {e['NN']:5.2f} (p {pn:4.2f}) | {e['VB']:5.2f} (p {pv:4.2f}) | {e['AD']:5.2f} (p {pa:4.2f})")


if __name__ == "__main__":
    main()
