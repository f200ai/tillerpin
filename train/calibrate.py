"""CPU temperature fitting, bound to one artifact/head and calibration-only data (AC-33)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tillerpin.tasks import ContractError, digest, require_sha, task

IDENTITY_FIELDS = {"modelSha256", "tokenizerSha256", "taskSchema", "labelOrder", "calibrationManifestSha256"}
FIT_FIELDS = IDENTITY_FIELDS | {"temperature", "n", "method", "metricsBefore", "metricsAfter"}


def validate_identity(identity: dict) -> None:
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise ContractError("calibration identity fields differ")
    for field in ("modelSha256", "tokenizerSha256", "calibrationManifestSha256"):
        require_sha(identity[field], field)
    schema = identity["taskSchema"]
    if not isinstance(schema, dict) or set(schema) != {"task", "labelSchemaVersion"}:
        raise ContractError("task schema differs")
    spec = task(schema["task"])
    if schema["labelSchemaVersion"] != spec.label_schema_version or identity["labelOrder"] != list(spec.labels):
        raise ContractError("calibration label schema or order differs")


def probabilities(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ContractError("temperature must be positive and finite")
    if not logits or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in logits
    ):
        raise ContractError("non-finite or empty model output")
    maximum = max(logits)
    scaled = [(x - maximum) / temperature for x in logits]
    if not all(math.isfinite(x) for x in scaled):
        raise ContractError("non-finite scaled model output")
    weights = [math.exp(x) for x in scaled]
    total = sum(weights)
    return [x / total for x in weights]


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    losses = []
    for row, label in zip(logits, labels, strict=True):
        maximum = max(row)
        scaled = [(x - maximum) / temperature for x in row]
        if not all(math.isfinite(x) for x in scaled):
            raise ContractError("non-finite scaled calibration logits")
        # Log-sum-exp preserves NLL even when the target probability underflows.
        losses.append(math.log(sum(math.exp(x) for x in scaled)) - scaled[label])
    loss = sum(x / len(losses) for x in losses)
    if not math.isfinite(loss):
        raise ContractError("non-finite calibration objective")
    return loss


def metrics(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> dict:
    probs = [probabilities(row, temperature) for row in logits]
    confidence = [max(p) for p in probs]
    correct = [int(p.index(max(p)) == y) for p, y in zip(probs, labels, strict=True)]
    bins = [[] for _ in range(10)]
    for c, hit in zip(confidence, correct, strict=True):
        bins[min(int(c * 10), 9)].append((c, hit))
    n = len(labels)
    ece = sum(abs(sum(c - hit for c, hit in b)) for b in bins) / n
    brier = sum(sum((p[j] - int(j == y)) ** 2 for j in range(len(p))) for p, y in zip(probs, labels, strict=True)) / n
    return {
        "nll": _nll(logits, labels, temperature),
        "brier": brier,
        "ece10": ece,
        "confidenceAccuracyGap": (sum(confidence) - sum(correct)) / n,
    }


def fit(
    logits: list[list[float]],
    labels: list[int],
    *,
    identity: dict,
    split: str,
    forbidden_pair_hashes: set[str],
    pair_hashes: list[str],
) -> dict:
    validate_identity(identity)
    if split != "calibration":
        raise ContractError("temperature fits require calibration rows")
    if not logits or len(logits) != len(labels) or len(pair_hashes) != len(labels):
        raise ContractError("empty or misaligned calibration rows")
    if len(set(pair_hashes)) != len(pair_hashes):
        raise ContractError("duplicate calibration row hashes")
    for sha in pair_hashes:
        require_sha(sha, "pairSha256")
    if set(pair_hashes) & forbidden_pair_hashes:
        raise ContractError("calibration overlaps training or selection")
    width = len(identity["labelOrder"])
    for row, y in zip(logits, labels, strict=True):
        if len(row) != width or isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < width:
            raise ContractError("calibration shape or target differs")
        probabilities(row, math.exp(-8.0))
    # Convex NLL in inverse temperature. Search log(T), with fixed, documented bounds.
    lo, hi = -8.0, 8.0
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(96):
        left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if _nll(logits, labels, math.exp(left)) <= _nll(logits, labels, math.exp(right)):
            hi = right
        else:
            lo = left
    temperature = math.exp((lo + hi) / 2)
    if _nll(logits, labels, temperature) > _nll(logits, labels, 1.0):
        temperature = 1.0
    record = {
        **identity,
        "temperature": temperature,
        "n": len(labels),
        "method": "temperature-nll-log-golden/1;logT=[-8,8]",
        "metricsBefore": metrics(logits, labels, 1.0),
        "metricsAfter": metrics(logits, labels, temperature),
    }
    applicable(record, identity)
    return record


def applicable(record: dict, identity: dict) -> str:
    """Return a fit identity only when every bound hash/schema matches and T is usable."""
    validate_identity(identity)
    if not isinstance(record, dict) or set(record) != FIT_FIELDS:
        raise ContractError("calibration record fields differ")
    if {k: record[k] for k in IDENTITY_FIELDS} != identity:
        raise ContractError("stale calibration fit")
    probabilities([0.0], record["temperature"])
    if isinstance(record["n"], bool) or not isinstance(record["n"], int) or record["n"] <= 0 or not record["method"]:
        raise ContractError("invalid fit provenance")
    for field in ("metricsBefore", "metricsAfter"):
        values = record[field]
        if not isinstance(values, dict) or set(values) != {"nll", "brier", "ece10", "confidenceAccuracyGap"}:
            raise ContractError("invalid calibration metrics")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
            raise ContractError("non-finite calibration metrics")
    return digest(record)
