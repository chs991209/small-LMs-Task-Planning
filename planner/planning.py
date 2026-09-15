"""Task planners (Strategy pattern).

A `Planner` turns a filled prompt into action steps. `OpenAIPlanner` is the LLM
baseline; a future `GPT2Planner` (local fine-tuned small LM) can drop in without
changing callers. `make_planner` is the factory.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import config


@dataclass
class PlanResult:
    text: str
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})


def build_prompt(domain: config.DomainConfig, objects, command: str) -> str:
    """Fill the domain's action-steps template. Fixed-object domains take (objects, command)."""
    template = domain.read_template()
    return template.format(objects, command) if domain.fixed_objects else template.format(command)


class Planner(ABC):
    """Strategy: a backend that generates action steps for a prompt."""
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> PlanResult:
        ...


class OpenAIPlanner(Planner):
    def __init__(self, model: str = "gpt-5.4-nano", client=None):
        self.model = model
        self.name = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = config.openai_client()
        return self._client

    def generate(self, prompt: str) -> PlanResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        u = resp.usage
        return PlanResult(
            text=resp.choices[0].message.content.strip(),
            usage={"input": u.prompt_tokens, "output": u.completion_tokens, "total": u.total_tokens},
        )


_REGISTRY = {"openai": OpenAIPlanner}


def make_planner(kind: str = "openai", **kwargs) -> Planner:
    """Factory: build a planner by kind. Extend _REGISTRY for new backends (e.g. gpt2)."""
    if kind not in _REGISTRY:
        raise SystemExit(f"Unknown planner {kind!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)
