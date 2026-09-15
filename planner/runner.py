"""Generation runs (Template Method pattern).

`GenerationRun.execute()` fixes the skeleton — iterate inputs, plan each, build a
record, assemble one run document, store it. Subclasses fill the variable parts:
`BaselineRun` pulls held-out dataset commands; `FigureSampleRun` uses fixed inputs.
"""
import json
from abc import ABC, abstractmethod

import planning
import stores


class GenerationRun(ABC):
    def __init__(self, domain, planner, store, task_name=None):
        self.domain = domain
        self.planner = planner
        self.store = store
        self.task_name = task_name or f"{domain.name}_{planner.name}"

    # --- template method (do not override) ---
    def execute(self):
        records, tin, tout = [], 0, 0
        for objects, command, extra in self.iter_inputs():
            prompt = planning.build_prompt(self.domain, objects, command)
            result = self.planner.generate(prompt)
            tin += result.usage["input"]
            tout += result.usage["output"]
            rec = self.build_record(objects, command, result, extra)
            records.append(rec)
            self.on_record(rec)
        doc = self.build_document(records, tin, tout)
        doc_id = stores.timestamped_id(self.task_name)
        self.store.save(doc_id, doc)
        return doc_id, doc

    # --- variable steps ---
    @abstractmethod
    def iter_inputs(self):
        """Yield (objects, command, extra_dict) tuples."""

    def build_record(self, objects, command, result, extra):
        return {
            "command": command,
            "objects_on_table": objects,
            "template_name": f"action_steps_{self.domain.name}",
            "model_output": result.text,
            "usage": result.usage,
            "executed_at": stores.now_local(),
            **(extra or {}),
        }

    def build_document(self, records, tin, tout):
        return {
            "task_name": self.task_name,
            "model": self.planner.name,
            "domain": self.domain.name,
            "n": len(records),
            "template_name": f"action_steps_{self.domain.name}",
            "template": self.domain.read_template(),
            "token_usage": {"input": tin, "output": tout, "total": tin + tout},
            "executed_at": stores.now_local(),
            "results": records,
        }

    def on_record(self, rec):  # hook
        u = rec["usage"]
        print(f"  in={u['input']} out={u['output']}  {rec['command']}")


class BaselineRun(GenerationRun):
    """Generates plans for the last `n` (held-out) commands of a COST dataset."""
    def __init__(self, domain, planner, store, n=10):
        super().__init__(domain, planner, store)
        self.n = n
        self._data = json.loads(domain.dataset_path.read_text())

    def iter_inputs(self):
        total = len(self._data["high_instructions"])
        for i in range(total - self.n, total):
            yield (
                self._data["objects_on_table"][i],
                self._data["high_instructions"][i],
                {"index": i, "ground_truth_steps": self._data["steps"][i]},
            )


class FigureSampleRun(GenerationRun):
    """Generates plans for fixed inputs (e.g. the paper's Fig. 8 / Fig. 9 samples)."""
    def __init__(self, domain, planner, store, samples):
        super().__init__(domain, planner, store, task_name=f"fig89_{domain.name}_{planner.name}")
        self.samples = samples

    def iter_inputs(self):
        for s in self.samples:
            yield (s["objects"], s["command"], {"figure": s.get("label")})
