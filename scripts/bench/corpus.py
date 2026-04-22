from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GoldTriple:
    subject: str
    predicate: str
    object: str

    def key(self) -> str:
        return (
            f"{self.subject.strip().lower()}||"
            f"{self.predicate.strip().lower()}||"
            f"{self.object.strip().lower()}"
        )


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    text: str
    gold_triples: Sequence[GoldTriple]


class Corpus(Protocol):
    """Read-only Protocol. Members are properties so `frozen=True` dataclasses
    (which have read-only fields) can satisfy the contract without the
    mutable-attribute requirement that plain attribute declarations impose.
    """

    @property
    def id(self) -> str: ...

    @property
    def documents(self) -> Sequence[CorpusDocument]: ...

    @property
    def snapshot_hash(self) -> str: ...


@dataclass(frozen=True)
class JsonCorpus:
    """Concrete corpus loader backed by a JSON file.

    Format:
        {
          "id": "seed_tiny_v1",
          "documents": [
            {"id": "doc_01", "text": "...", "gold_triples": [[s, p, o], ...]}
          ]
        }

    Gold triples MUST be pre-canonicalized: lowercased, stripped, and lemmatized
    to match the output of sum_engine_internal.algorithms.syntactic_sieve.DeterministicSieve.
    Mismatches caused by un-lemmatized gold are counted as false negatives, not
    reconciled post-hoc. Honesty over flattery.
    """

    id: str
    documents: tuple[CorpusDocument, ...]
    snapshot_hash: str

    @classmethod
    def load(cls, path: Path) -> "JsonCorpus":
        if not path.exists():
            raise FileNotFoundError(f"corpus not found: {path}")
        raw = path.read_bytes()
        snapshot_hash = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw.decode("utf-8"))
        docs = tuple(
            CorpusDocument(
                id=d["id"],
                text=d["text"],
                gold_triples=tuple(
                    GoldTriple(subject=s, predicate=p, object=o)
                    for s, p, o in d["gold_triples"]
                ),
            )
            for d in data["documents"]
        )
        return cls(id=data["id"], documents=docs, snapshot_hash=snapshot_hash)
