import copy
import math

import pytest

from tillerpin.tasks import ContractError
from train.calibrate import applicable, fit, probabilities


def identity():
    return {
        "modelSha256": "a" * 64,
        "tokenizerSha256": "b" * 64,
        "taskSchema": {"task": "entailment_v1", "labelSchemaVersion": "1"},
        "labelOrder": ["entailment", "neutral", "contradiction"],
        "calibrationManifestSha256": "c" * 64,
    }


def fitted(**changes):
    args = dict(
        identity=identity(),
        split="calibration",
        forbidden_pair_hashes=set(),
        pair_hashes=["1" * 64, "2" * 64, "3" * 64],
    )
    args.update(changes)
    return fit([[8.0, 0.0, 0.0]] * 3, [0, 1, 2], **args)


def test_temperature_improves_nll_and_binds_artifact():
    record = fitted()
    assert math.isfinite(record["temperature"]) and record["temperature"] > 1
    assert record["metricsAfter"]["nll"] < record["metricsBefore"]["nll"]
    assert len(applicable(record, identity())) == 64
    for field in ("modelSha256", "tokenizerSha256", "calibrationManifestSha256"):
        with pytest.raises(ContractError, match="stale"):
            applicable(record, {**identity(), field: "f" * 64})


@pytest.mark.parametrize("temperature", [None, True, 0, -1, float("nan"), float("inf")])
def test_invalid_fit_is_not_calibrated(temperature):
    record = fitted()
    record["temperature"] = temperature
    with pytest.raises(ContractError):
        applicable(record, identity())


def test_schema_and_label_order_refused():
    changed = copy.deepcopy(identity())
    changed["labelOrder"].reverse()
    with pytest.raises(ContractError, match="order"):
        fitted(identity=changed)


@pytest.mark.parametrize("split", ["train", "selection", "test"])
def test_fit_never_uses_non_calibration_rows(split):
    with pytest.raises(ContractError, match="calibration rows"):
        fitted(split=split)


def test_fit_refuses_overlap_and_nonfinite_logits():
    with pytest.raises(ContractError, match="overlap"):
        fitted(forbidden_pair_hashes={"2" * 64})
    with pytest.raises(ContractError, match="duplicate"):
        fitted(pair_hashes=["1" * 64] * 3)
    with pytest.raises(ContractError):
        fit(
            [[float("nan"), 0.0, 0.0]],
            [0],
            identity=identity(),
            split="calibration",
            forbidden_pair_hashes=set(),
            pair_hashes=["1" * 64],
        )


def test_softmax_is_stable_and_directional_order_is_preserved():
    assert probabilities([1000, 1000, 1000]) == pytest.approx([1 / 3] * 3)
    assert probabilities([1000, 0, -1000])[0] == 1.0


@pytest.mark.parametrize("row", [[1e308, -1e308, 0], [1e308, 0, 0]])
def test_finite_logits_that_overflow_temperature_search_fail_closed(row):
    with pytest.raises(ContractError, match="non-finite"):
        fit([row], [1], identity=identity(), split="calibration", forbidden_pair_hashes=set(), pair_hashes=["1" * 64])
