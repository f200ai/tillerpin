# TILLERPIN source contracts

This source alpha is a developer toolkit for
closed-task label schemas, temperature calibration, experiment planning and
fail-closed checkpoint admission. It is narrower than the planned TILLERPIN
inference product. No trained classifier or measured model quality is included.

The code uses Python 3.12 or newer and the standard library. Tests use pytest.
From this directory, run:

```sh
python -m pip install -e '.[test]'
python -m pytest
python examples/calibration.py
```

The example uses synthetic logits to demonstrate calibration and artifact-identity
binding. It is not a model inference example and establishes no real-world accuracy.
The low-level `metrics` and `probabilities` helpers assume caller-validated inputs;
they are not a general-purpose untrusted-input boundary. The example supplies
finite logits explicitly. Temperature fitting requires a separate calibration split, rejects duplicate or
forbidden pair hashes, and refuses stale identity, reordered labels and non-finite
values. Callers remain responsible for genuine held-out data and provenance.

`train.recipe` validates experiment plans and selection records; it does not launch
compute. `train.backbones.safe_checkpoint` checks supplied local safe-tensor framing,
file hashes and required conversion evidence; it does not execute a pickle loader,
download a model, authorize a conversion or attest that an external reviewer approved
it. Full model identifiers/pins in this contract are references, not supplied artifacts.

The entailment schema is directional. Evidence-relation schema definitions describe
an intended task and do not mean that a head was trained. Use the explicit
`relation_outcome="not trained (class ambiguous)"` declaration when planning an
entailment-only artifact.

Not included: inference and HTTP serving, the compatibility adapter,
native training/export runtimes, trained weights, tokenizers, dataset text, annotation
queues, evaluation outputs, private infrastructure and private repository history.
There is no latency, calibration quality, security-decision accuracy or deployment
claim. Apache-2.0 covers this code only; external artifact rights remain separate.
