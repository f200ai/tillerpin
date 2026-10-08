"""Versioned directional task schemas and artifact identity (mission AC-30)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    label_schema_version: str = "1"

    def metadata(self) -> dict:
        return {"task": self.name, "labelOrder": list(self.labels), "labelSchemaVersion": self.label_schema_version}


TASKS: Mapping[str, Task] = MappingProxyType(
    {
        "entailment_v1": Task("entailment_v1", ("entailment", "neutral", "contradiction")),
        "evidence_relation_v1": Task("evidence_relation_v1", ("support", "contradict", "unrelated", "ambiguous")),
    }
)


def task(name: str) -> Task:
    try:
        return TASKS[name]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"unknown task {name!r}") from exc


def released_tasks(relation_outcome: str = "trained") -> tuple[Task, ...]:
    if relation_outcome == "trained":
        return tuple(TASKS.values())
    if re.fullmatch(r"not trained \(class (support|contradict|unrelated|ambiguous)\)", relation_outcome):
        return (task("entailment_v1"),)
    raise ContractError("relation outcome must be trained or an explicit AC-36 class decision")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_sha(value, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{field} must be a sha256")
    return value


def artifact_identity(metadata: dict) -> str:
    """Hash only a validated, explicit contract, including installed heads and calibration fits."""
    keys = {"modelSha256", "tokenizerSha256", "dataManifestSha256", "heads", "relationOutcome"}
    if not isinstance(metadata, dict) or set(metadata) != keys:
        raise ContractError("artifact identity fields differ")
    for field in ("modelSha256", "tokenizerSha256", "dataManifestSha256"):
        require_sha(metadata[field], field)
    installed = released_tasks(metadata["relationOutcome"])
    heads = metadata["heads"]
    if not isinstance(heads, list) or len(heads) != len(installed):
        raise ContractError("installed heads differ from release decision")
    for head, spec in zip(heads, installed, strict=True):
        if not isinstance(head, dict) or set(head) != {
            "task",
            "labelOrder",
            "labelSchemaVersion",
            "calibrationIdentity",
        }:
            raise ContractError("head identity fields differ")
        if {k: head[k] for k in spec.metadata()} != spec.metadata():
            raise ContractError("head schema or label order differs")
        if head["calibrationIdentity"] is not None:
            require_sha(head["calibrationIdentity"], "calibrationIdentity")
    return digest(metadata)


def verify_artifact(metadata: dict, expected_identity: str) -> None:
    require_sha(expected_identity, "artifactIdentity")
    if artifact_identity(metadata) != expected_identity:
        raise ContractError("artifact identity mismatch")
