"""Reproduce the paper's Fig. 8 (tabletop) & Fig. 9 (kitchen) samples with an LLM
planner, write a text file, and upload to Firestore. Thin wrapper over
runner.FigureSampleRun (one run per figure, since each figure is a different domain).
"""
import config
import stores
from planning import make_planner
from runner import FigureSampleRun

FIG8 = {"label": "Fig. 8", "domain": "tabletop",
        "objects": ["Yellow semicircle block", "Red circle block", "Red semicircle block", "Yellow bowl", "Red bowl"],
        "command": "Arrange semicircle blocks in the same colored bowl as the picked block."}
FIG9 = {"label": "Fig. 9", "domain": "kitchen",
        "objects": ["Cucumber", "Lettuce", "Tray", "Sink", "Knife", "Cup"],
        "command": "Slice the washed lettuce and place it on tray."}
OUT_PATH = "fig89_samples_gpt-5.4-nano.txt"


def main(model="gpt-5.4-nano", upload=True):
    planner = make_planner("openai", model=model)
    blocks = []
    for fig in (FIG8, FIG9):
        domain = config.get_domain(fig["domain"])
        store = stores.CompositeStore([stores.FirestoreStore()] if upload else [])
        _, doc = FigureSampleRun(domain, planner, store, samples=[fig]).execute()
        rec = doc["results"][0]
        blocks.append("\n".join([
            "=" * 72,
            f"{fig['label']} ({fig['domain']}) — {model}",
            f"Objects= {fig['objects']}",
            f"Command= {fig['command']}",
            "-" * 72,
            rec["model_output"],
            "",
        ]))
    header = ("Reproduction of the paper's Fig. 8 (tabletop) and Fig. 9 (kitchen) samples,\n"
              "generated with an LLM planner on the exact figure inputs (the fine-tuned\n"
              "GPT2-medium checkpoint from the paper is not available in this repo).\n\n")
    with open(OUT_PATH, "w") as f:
        f.write(header + "\n".join(blocks))
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
