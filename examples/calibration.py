"""Synthetic calibration example; no model inference or quality claim."""
from train.calibrate import applicable, fit, probabilities

identity = {
    "modelSha256": "a" * 64,
    "tokenizerSha256": "b" * 64,
    "taskSchema": {"task": "entailment_v1", "labelSchemaVersion": "1"},
    "labelOrder": ["entailment", "neutral", "contradiction"],
    "calibrationManifestSha256": "c" * 64,
}
record = fit(
    [[8.0, 0.0, 0.0]] * 3,
    [0, 1, 2],
    identity=identity,
    split="calibration",
    forbidden_pair_hashes=set(),
    pair_hashes=["1" * 64, "2" * 64, "3" * 64],
)
print("Synthetic example; calibration identity:", applicable(record, identity))
print("Temperature:", record["temperature"])
print("Probabilities:", probabilities([8.0, 0.0, 0.0], record["temperature"]))
