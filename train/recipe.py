"""Frozen experiment planning and selection; no model/cloud launcher is embedded (AC-32)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from tillerpin.tasks import ContractError, digest, released_tasks, require_sha
from train.backbones import BACKBONES

SEEDS = (13, 21, 42, 87, 1234)
REQUIRED_ARMS = ("A", "B", "E")
WAVES = ("W0", "W1", "W2", "W3")
FIELDS = {
    "schema",
    "id",
    "dataManifestSha256",
    "splitSha256",
    "relationOutcome",
    "arms",
    "seeds",
    "protocol",
    "floors",
    "waves",
}
PROTOCOL = {
    "maxTokens": 512,
    "loss": "cross_entropy",
    "learningRates": [1e-5, 2e-5, 3e-5],
    "effectiveBatches": [16, 32],
    "maxEpochs": 4,
    "warmupShare": 0.06,
    "weightDecay": 0.01,
    "schedule": "linear",
    "selection": "five-seed-mean-macro-f1,nll,cpu-latency;checkpoint-macro-f1,nll",
}


def positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ContractError(f"{name} must be positive and finite")
    return float(value)


def validate(experiment: dict, frozen_sha256: str) -> str:
    require_sha(frozen_sha256, "frozenExperimentSha256")
    if (
        not isinstance(experiment, dict)
        or set(experiment) != FIELDS
        or experiment["schema"] != "tillerpin-experiment/1"
    ):
        raise ContractError("experiment fields or schema differ")
    actual = digest(experiment)
    if actual != frozen_sha256:
        raise ContractError("experiment differs from its frozen identity")
    if not isinstance(experiment["id"], str) or not experiment["id"].strip():
        raise ContractError("experiment id is absent")
    require_sha(experiment["dataManifestSha256"], "dataManifestSha256")
    splits = experiment["splitSha256"]
    if not isinstance(splits, dict) or set(splits) != {"train", "selection", "calibration", "test"}:
        raise ContractError("every split must freeze before selection")
    for name, sha in splits.items():
        require_sha(sha, name)
    if len(set(splits.values())) != len(splits):
        raise ContractError("split identities are not disjoint")
    if experiment["seeds"] != list(SEEDS) or experiment["protocol"] != PROTOCOL:
        raise ContractError("experiment protocol differs from the five-seed declaration")
    arms = experiment["arms"]
    if not isinstance(arms, dict) or set(arms) != set(REQUIRED_ARMS):
        raise ContractError("required arms A, B and E must receive equal trials")
    for name, arm in arms.items():
        spec = BACKBONES[name]
        if not isinstance(arm, dict) or set(arm) != {
            "backbone",
            "revision",
            "safeArtifactSha256",
            "conversionRecordSha256",
        }:
            raise ContractError("arm fields differ")
        revision = arm["revision"]
        if (
            arm["backbone"] != spec.name
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or not revision.startswith(spec.revision_prefix)
        ):
            raise ContractError("arm backbone revision differs")
        for field in ("safeArtifactSha256", "conversionRecordSha256"):
            if arm[field] is not None:
                require_sha(arm[field], field)
    tasks = released_tasks(experiment["relationOutcome"])
    floors = experiment["floors"]
    if not isinstance(floors, dict) or set(floors) != {t.name for t in tasks}:
        raise ContractError("task quality floors differ")
    for name, floor in floors.items():
        minimum = 0.85 if name == "entailment_v1" else 0.50
        if positive(floor, "floor") < minimum or floor > 1:
            raise ContractError("quality floor is below the product-owner minimum")
    waves = experiment["waves"]
    if not isinstance(waves, dict) or set(waves) != {"W0", "W1", "W2"}:
        raise ContractError("required waves W0/W1/W2 differ; conditional F needs its own reviewed manifest")
    for wave, record in waves.items():
        if not isinstance(record, dict) or set(record) != {"plannedGpuHours", "plannedUsd", "measurementSha256"}:
            raise ContractError("wave plan fields differ")
        positive(record["plannedGpuHours"], "plannedGpuHours")
        positive(record["plannedUsd"], "plannedUsd")
        if record["measurementSha256"] is not None:
            require_sha(record["measurementSha256"], "measurementSha256")
        if wave != "W0" and record["measurementSha256"] is None:
            # Draft W1/W2 estimates may be documented, but plan_wave will refuse them.
            continue
    return actual


def plan_wave(experiment: dict, frozen_sha256: str, wave: str, budget: dict | None) -> dict:
    sha = validate(experiment, frozen_sha256)
    if wave not in experiment["waves"]:
        raise ContractError("wave is not preregistered")
    record = experiment["waves"][wave]
    reasons = []
    if budget is None:
        reasons.append("no approved wave budget")
    elif not isinstance(budget, dict) or set(budget) != {
        "wave",
        "experimentSha256",
        "maxUsd",
        "operatorApprovalEvidence",
        "measurementSha256",
    }:
        reasons.append("budget fields differ")
    else:
        if (
            budget["wave"] != wave
            or budget["experimentSha256"] != sha
            or not isinstance(budget["operatorApprovalEvidence"], str)
            or not budget["operatorApprovalEvidence"].strip()
        ):
            reasons.append("budget is not bound to this approved experiment wave")
        if positive(budget["maxUsd"], "maxUsd") < record["plannedUsd"]:
            reasons.append("planned cost exceeds the wave cap")
        if budget["measurementSha256"] != record["measurementSha256"]:
            reasons.append("budget measurement identity differs")
    if wave != "W0" and record["measurementSha256"] is None:
        reasons.append("later-wave cap requires W0 measurements and fresh approval")
    arms = {}
    for name, arm in experiment["arms"].items():
        safe = arm["safeArtifactSha256"] is not None and (
            not BACKBONES[name].conversion_required or arm["conversionRecordSha256"] is not None
        )
        arms[name] = "ready for operator" if safe else "not run (no safe artifact)"
        if not safe:
            reasons.append(f"arm {name}: no safe artifact or reviewed conversion")
    return {
        "experimentSha256": sha,
        "wave": wave,
        **record,
        "arms": arms,
        "verdict": "refused-to-start" if reasons else "ready-for-operator-handoff",
        "reasons": reasons,
        "launchImplemented": False,
    }


def select(experiment: dict, frozen_sha256: str, results: list[dict]) -> dict:
    """Select from selection-split aggregates only; no data/test file is opened."""
    sha = validate(experiment, frozen_sha256)
    names = set(experiment["floors"])
    grouped = {arm: [] for arm in REQUIRED_ARMS}
    seen = set()
    for row in results:
        if not isinstance(row, dict) or set(row) != {
            "experimentSha256",
            "arm",
            "seed",
            "split",
            "macroF1",
            "nll",
            "cpuLatencyMs",
            "checkpointSha256",
        }:
            raise ContractError("selection result fields differ")
        if (
            row["experimentSha256"] != sha
            or row["split"] != "selection"
            or row["arm"] not in grouped
            or row["seed"] not in SEEDS
        ):
            raise ContractError("result is not from the frozen selection experiment")
        key = (row["arm"], row["seed"])
        if key in seen:
            raise ContractError("duplicate arm/seed result")
        seen.add(key)
        require_sha(row["checkpointSha256"], "checkpointSha256")
        if not isinstance(row["macroF1"], dict) or set(row["macroF1"]) != names:
            raise ContractError("head metrics differ")
        for value in row["macroF1"].values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ContractError("invalid macro-F1")
        for field in ("nll", "cpuLatencyMs"):
            positive(row[field], field)
        grouped[row["arm"]].append(row)
    if seen != {(arm, seed) for arm in REQUIRED_ARMS for seed in SEEDS}:
        raise ContractError("selection needs all five seeds for each required arm")
    qualifying = {
        arm: [r for r in rows if all(r["macroF1"][n] >= f for n, f in experiment["floors"].items())]
        for arm, rows in grouped.items()
    }
    candidates = [arm for arm in REQUIRED_ARMS if qualifying[arm]]
    if not candidates:
        return {"verdict": "no release candidate", "experimentSha256": sha}

    def score(row):
        return mean(row["macroF1"].values())

    winner = min(
        candidates,
        key=lambda a: (
            -mean(score(r) for r in grouped[a]),
            mean(r["nll"] for r in grouped[a]),
            mean(r["cpuLatencyMs"] for r in grouped[a]),
            a,
        ),
    )
    checkpoint = min(qualifying[winner], key=lambda r: (-score(r), r["nll"], r["seed"]))
    return {
        "verdict": "selected",
        "experimentSha256": sha,
        "arm": winner,
        "seed": checkpoint["seed"],
        "checkpointSha256": checkpoint["checkpointSha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--wave", choices=WAVES, default="W0")
    parser.add_argument("--budget", type=Path)
    args = parser.parse_args(argv)
    try:
        result = plan_wave(
            json.loads(args.experiment.read_text()),
            args.frozen_sha256,
            args.wave,
            json.loads(args.budget.read_text()) if args.budget else None,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"verdict": "refused-to-start", "reason": str(exc), "launchImplemented": False}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "ready-for-operator-handoff" else 2


if __name__ == "__main__":
    raise SystemExit(main())
