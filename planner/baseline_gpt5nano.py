"""Baseline entrypoint: run an LLM planner (default gpt-5.4-nano) over held-out
COST commands, saving locally + to Firestore. Thin wrapper over runner.BaselineRun.
Paper: Choi & Ahn, arXiv 2404.03891.
"""
import argparse

import config
import stores
from planning import make_planner
from runner import BaselineRun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(config.DOMAINS), default="tabletop")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--n", type=int, default=10, help="number of held-out examples")
    ap.add_argument("--out", default=None, help="local json path")
    ap.add_argument("--no-upload", action="store_true", help="skip Firestore upload")
    args = ap.parse_args()

    domain = config.get_domain(args.domain)
    planner = make_planner("openai", model=args.model)
    out_path = args.out or f"baseline_{args.domain}_{args.model}.json"

    backends = [stores.LocalJsonStore(out_path)]
    if not args.no_upload:
        backends.append(stores.FirestoreStore())
    store = stores.CompositeStore(backends)

    doc_id, doc = BaselineRun(domain, planner, store, n=args.n).execute()

    u = doc["token_usage"]
    print(f"\n{doc['n']} plans | tokens in={u['input']} out={u['output']} total={u['total']}")
    print(f"doc id: {doc_id}")


if __name__ == "__main__":
    main()
