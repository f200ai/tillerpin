import copy
import json

import pytest

from tillerpin.tasks import ContractError, digest
from train.backbones import BACKBONES, safe_checkpoint
from train.recipe import PROTOCOL, REQUIRED_ARMS, SEEDS, main, plan_wave, select, validate


def safe_fixture(path):
    import hashlib
    import json
    import struct

    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    raw = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    path.write_bytes(raw)
    return {path.name: hashlib.sha256(raw).hexdigest()}


def experiment():
    return {
        "schema": "tillerpin-experiment/1",
        "id": "fixture",
        "dataManifestSha256": "d" * 64,
        "splitSha256": {name: str(i + 1) * 64 for i, name in enumerate(("train", "selection", "calibration", "test"))},
        "relationOutcome": "not trained (class ambiguous)",
        "arms": {
            name: {
                "backbone": BACKBONES[name].name,
                "revision": BACKBONES[name].revision_prefix + "0" * 32,
                "safeArtifactSha256": "a" * 64,
                "conversionRecordSha256": "b" * 64 if name != "A" else None,
            }
            for name in REQUIRED_ARMS
        },
        "seeds": list(SEEDS),
        "protocol": copy.deepcopy(PROTOCOL),
        "floors": {"entailment_v1": 0.85},
        "waves": {
            name: {"plannedGpuHours": 2, "plannedUsd": 5, "measurementSha256": None} for name in ("W0", "W1", "W2")
        },
    }


def budget(exp, wave="W0"):
    return {
        "wave": wave,
        "experimentSha256": digest(exp),
        "maxUsd": 15,
        "operatorApprovalEvidence": "synthetic-operator-approval",
        "measurementSha256": exp["waves"][wave]["measurementSha256"],
    }


def results(exp):
    return [
        {
            "experimentSha256": digest(exp),
            "arm": arm,
            "seed": seed,
            "split": "selection",
            "macroF1": {"entailment_v1": {"A": 0.90, "B": 0.91, "E": 0.89}[arm]},
            "nll": 0.3,
            "cpuLatencyMs": 50,
            "checkpointSha256": str(i + 1) * 64,
        }
        for i, arm in enumerate(REQUIRED_ARMS)
        for seed in SEEDS
    ]


def test_no_budget_refuses_and_never_launches(tmp_path, capsys):
    exp = experiment()
    planned = plan_wave(exp, digest(exp), "W0", None)
    assert planned["verdict"] == "refused-to-start" and planned["plannedGpuHours"] == 2
    assert not planned["launchImplemented"]
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(exp))
    assert main([str(path), "--frozen-sha256", digest(exp)]) == 2
    assert "refused-to-start" in capsys.readouterr().out


def test_approved_budget_only_prepares_operator_handoff():
    exp = experiment()
    assert plan_wave(exp, digest(exp), "W0", budget(exp))["verdict"] == "ready-for-operator-handoff"
    cap = {**budget(exp), "maxUsd": 4}
    assert "planned cost exceeds" in " ".join(plan_wave(exp, digest(exp), "W0", cap)["reasons"])
    assert plan_wave(exp, digest(exp), "W1", budget(exp, "W1"))["verdict"] == "refused-to-start"
    exp["waves"]["W1"]["measurementSha256"] = "f" * 64
    assert plan_wave(exp, digest(exp), "W1", budget(exp, "W1"))["verdict"] == "ready-for-operator-handoff"


def test_missing_safe_conversion_prevents_arm_start():
    exp = experiment()
    exp["arms"]["E"]["conversionRecordSha256"] = None
    plan = plan_wave(exp, digest(exp), "W0", budget(exp))
    assert plan["arms"]["E"] == "not run (no safe artifact)"
    assert plan["verdict"] == "refused-to-start"


@pytest.mark.parametrize("field", ["dataManifestSha256", "splitSha256", "protocol", "arms", "floors", "seeds"])
def test_missing_manifest_fields_refuse(field):
    exp = experiment()
    del exp[field]
    with pytest.raises(ContractError):
        validate(exp, digest(exp))


def test_changed_floors_or_protocol_refuse_after_freeze():
    exp = experiment()
    frozen = digest(exp)
    exp["floors"]["entailment_v1"] = 0.84
    with pytest.raises(ContractError, match="frozen"):
        validate(exp, frozen)
    with pytest.raises(ContractError, match="minimum"):
        validate(exp, digest(exp))
    exp = experiment()
    del exp["arms"]["E"]
    with pytest.raises(ContractError, match="required arms"):
        validate(exp, digest(exp))


def test_selection_uses_five_seed_arm_mean_then_checkpoint_floors():
    exp = experiment()
    rows = results(exp)
    outcome = select(exp, digest(exp), rows)
    assert outcome["arm"] == "B" and outcome["seed"] == 13
    for row in rows:
        row["macroF1"]["entailment_v1"] = 0.84
    assert select(exp, digest(exp), rows)["verdict"] == "no release candidate"


def test_test_results_and_missing_or_duplicate_seed_refused():
    exp = experiment()
    rows = results(exp)
    with pytest.raises(ContractError, match="five seeds"):
        select(exp, digest(exp), rows[:-1])
    with pytest.raises(ContractError, match="duplicate"):
        select(exp, digest(exp), rows + [rows[0]])
    rows[0]["split"] = "test"
    with pytest.raises(ContractError, match="frozen selection"):
        select(exp, digest(exp), rows)


def test_checkpoint_preflight_refuses_pickle_and_missing_conversion(tmp_path):
    artifacts = safe_fixture(tmp_path / "model.safetensors")
    assert safe_checkpoint(tmp_path, "A", revision="45bb4654" + "0" * 32, artifacts=artifacts)
    with pytest.raises(ContractError, match="conversion"):
        safe_checkpoint(tmp_path, "B", revision="64a8c8ea" + "0" * 32, artifacts=artifacts)
    (tmp_path / "weights.bin").write_bytes(b"never load")
    with pytest.raises(ContractError, match="unsafe"):
        safe_checkpoint(tmp_path, "A", revision="45bb4654" + "0" * 32, artifacts=artifacts)


def test_checkpoint_directory_symlink_and_non_string_revision_are_rejected(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    artifacts = safe_fixture(weights / "model.safetensors")
    link = tmp_path / "link"
    link.symlink_to(weights, target_is_directory=True)
    with pytest.raises(ContractError, match="directory"):
        safe_checkpoint(link, "A", revision="45bb4654" + "0" * 32, artifacts=artifacts)
    with pytest.raises(ContractError, match="revision"):
        safe_checkpoint(weights, "A", revision=None, artifacts=artifacts)


def test_checkpoint_bytes_and_reviewed_conversion_are_hash_bound(tmp_path):
    import hashlib

    path = tmp_path / "model.safetensors"
    artifacts = safe_fixture(path)
    revision = "64a8c8ea" + "0" * 32
    record = {
        "schema": "backbone-conversion/1",
        "revision": revision,
        "reviewEvidence": "synthetic approval fixture",
        "sourceSha256": "a" * 64,
        "outputs": artifacts,
        "tensorEqualityVerificationSha256": "b" * 64,
    }
    args = dict(
        revision=revision,
        artifacts=artifacts,
        conversion=record,
        conversion_sha256=digest(record),
        approved_verification_sha256="b" * 64,
    )
    assert safe_checkpoint(tmp_path, "B", **args)
    with pytest.raises(ContractError, match="record identity"):
        safe_checkpoint(tmp_path, "B", **{**args, "conversion_sha256": "c" * 64})
    with pytest.raises(ContractError, match="verification"):
        safe_checkpoint(tmp_path, "B", **{**args, "approved_verification_sha256": "c" * 64})
    path.write_bytes(b"not safetensors")
    with pytest.raises(ContractError, match="artifact identity"):
        safe_checkpoint(tmp_path, "B", **args)
    with pytest.raises(ContractError, match="safetensors"):
        safe_checkpoint(
            tmp_path,
            "A",
            revision="45bb4654" + "0" * 32,
            artifacts={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
        )


@pytest.mark.parametrize(
    "header",
    [
        {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
        {"x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}},
        {"x": {"dtype": "OBJECT", "shape": [1], "data_offsets": [0, 4]}},
        {"__metadata__": {"object": ["not a string"]}},
    ],
)
def test_malformed_safetensors_header_is_rejected_even_when_hash_is_pinned(tmp_path, header):
    import hashlib
    import json
    import struct

    raw_header = json.dumps(header).encode()
    raw = struct.pack("<Q", len(raw_header)) + raw_header + b"1234"
    path = tmp_path / "model.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ContractError, match="safetensors"):
        safe_checkpoint(
            tmp_path, "A", revision="45bb4654" + "0" * 32, artifacts={path.name: hashlib.sha256(raw).hexdigest()}
        )
