"""Result storage (Repository pattern).

`ResultStore` abstracts where a run's document goes; `CompositeStore` fans out to
several backends (e.g. local JSON + Firestore). `upload_dataset` bulk-loads the
paper's ground-truth COST pairs as individual documents.
"""
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import config

COLLECTION = "robotics_tasks"


def timestamped_id(task_name: str) -> str:
    """Doc id like 20260916022302_task_name (YYYYMMDDHHMMSS_<task_name>)."""
    return f"{datetime.now().astimezone():%Y%m%d%H%M%S}_{task_name}"


def now_local() -> str:
    return datetime.now().astimezone().isoformat()


class ResultStore(ABC):
    @abstractmethod
    def save(self, doc_id: str, data: dict) -> str:
        ...


class LocalJsonStore(ResultStore):
    """Writes the run document to a fixed local JSON path (latest-run snapshot)."""
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, doc_id: str, data: dict) -> str:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return str(self.path)


class FirestoreStore(ResultStore):
    """Writes the run document to a Firestore collection under `doc_id`."""
    def __init__(self, collection: str = COLLECTION, client=None):
        self.collection = collection
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = config.firestore_client()
        return self._client

    def save(self, doc_id: str, data: dict) -> str:
        self.client.collection(self.collection).document(doc_id).set(data)
        return doc_id


class CompositeStore(ResultStore):
    """Fans out a save to several stores; a failing backend warns but doesn't abort."""
    def __init__(self, stores):
        self.stores = list(stores)

    def save(self, doc_id: str, data: dict) -> str:
        for s in self.stores:
            try:
                where = s.save(doc_id, data)
                print(f"  stored -> {type(s).__name__}: {where}")
            except Exception as e:
                print(f"  [warn] {type(s).__name__} save skipped: {e}")
        return doc_id


def upload_dataset(domain_name: str, limit: int | None = None,
                   store: FirestoreStore | None = None) -> int:
    """Store each COST command-steps pair as its own document (batched writes)."""
    domain = config.get_domain(domain_name)
    d = json.loads(domain.dataset_path.read_text())
    n = len(d["high_instructions"])
    if limit is not None:
        n = min(n, limit)
    store = store or FirestoreStore()
    client = store.client
    BATCH = 400  # Firestore max is 500 ops/batch
    written = 0
    for start in range(0, n, BATCH):
        batch = client.batch()
        for i in range(start, min(start + BATCH, n)):
            ref = client.collection(store.collection).document(f"{domain.name}_{i:06d}")
            batch.set(ref, {
                "domain": domain.name,
                "index": i,
                "command": d["high_instructions"][i],
                "objects_on_table": d["objects_on_table"][i],
                "used_objects": d["used_objects"][i],
                "steps": d["steps"][i],
                "source": "COST dataset (paper ground-truth)",
                "executed_at": now_local(),
            })
        batch.commit()
        written = min(start + BATCH, n)
        print(f"  ...{written}/{n}")
    return written
