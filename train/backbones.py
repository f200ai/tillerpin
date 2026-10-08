"""Architecture pins, deliberately separate from any untrusted checkpoint loader."""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from tillerpin.tasks import ContractError, digest, require_sha


@dataclass(frozen=True)
class Backbone:
    name: str
    revision_prefix: str
    conversion_required: bool
    attention: str


BACKBONES = {
    "A": Backbone("answerdotai/ModernBERT-large", "45bb4654", False, "flash_attention_2"),
    "B": Backbone("microsoft/deberta-v3-large", "64a8c8ea", True, "sdpa"),
    "E": Backbone("jhu-clsp/ettin-encoder-400m", "7662476d", True, "flash_attention_2"),
    "F": Backbone("jhu-clsp/ettin-encoder-1b", "befd76be", True, "flash_attention_2"),
}


def _safe_header(path: Path) -> None:
    """Validate safetensors framing, tensor sizes and contiguous bounds without executing a loader."""

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ContractError("duplicate safetensors header key")
            obj[key] = value
        return obj

    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ContractError("invalid safetensors framing")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= min(1 << 20, size - 8):
                raise ContractError("invalid safetensors header length")
            header = json.loads(stream.read(length), object_pairs_hook=unique)
        if not isinstance(header, dict):
            raise ContractError("invalid safetensors header")
        ranges = []
        for name, tensor in header.items():
            if name == "__metadata__":
                if not isinstance(tensor, dict) or any(
                    not isinstance(k, str) or not isinstance(v, str) for k, v in tensor.items()
                ):
                    raise ContractError("invalid safetensors metadata")
                continue
            if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
                raise ContractError("invalid safetensors tensor")
            shape, offsets = tensor["shape"], tensor["data_offsets"]
            if (
                not isinstance(tensor["dtype"], str)
                or tensor["dtype"] not in sizes
                or not isinstance(shape, list)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in offsets)
            ):
                raise ContractError("invalid safetensors shape or offsets")
            start, end = offsets
            if end < start or end - start != math.prod(shape) * sizes[tensor["dtype"]]:
                raise ContractError("invalid safetensors tensor byte count")
            ranges.append((start, end))
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ContractError("safetensors data overlap or gap")
            cursor = end
        if not ranges or cursor != size - 8 - length:
            raise ContractError("safetensors data bounds differ")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ContractError("invalid safetensors artifact") from exc


def safe_checkpoint(
    directory: Path,
    arm: str,
    *,
    revision: str,
    artifacts: dict[str, str],
    conversion: dict | None = None,
    conversion_sha256: str | None = None,
    approved_verification_sha256: str | None = None,
) -> tuple[Path, ...]:
    if arm not in BACKBONES:
        raise ContractError("unadmitted arm")
    spec = BACKBONES[arm]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
        or not revision.startswith(spec.revision_prefix)
    ):
        raise ContractError("backbone requires its resolved immutable revision")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError("unsafe checkpoint directory")
    files = tuple(directory.rglob("*"))
    if any(
        p.is_symlink() or p.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib"}
        for p in files
    ):
        raise ContractError("unsafe checkpoint artifact")
    weights = tuple(sorted(p for p in files if p.is_file() and p.suffix == ".safetensors"))
    if not weights:
        raise ContractError("no safe artifact")
    if not isinstance(artifacts, dict) or set(artifacts) != {p.relative_to(directory).as_posix() for p in weights}:
        raise ContractError("pinned checkpoint artifact manifest differs")
    for path in weights:
        name = path.relative_to(directory).as_posix()
        require_sha(artifacts[name], "checkpointSha256")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                sha.update(chunk)
        if sha.hexdigest() != artifacts[name]:
            raise ContractError("checkpoint artifact identity mismatch")
        _safe_header(path)
    if spec.conversion_required:
        fields = {"schema", "revision", "reviewEvidence", "sourceSha256", "outputs", "tensorEqualityVerificationSha256"}
        if (
            not isinstance(conversion, dict)
            or set(conversion) != fields
            or conversion["schema"] != "backbone-conversion/1"
            or conversion["revision"] != revision
            or not isinstance(conversion["reviewEvidence"], str)
            or not conversion["reviewEvidence"].strip()
            or conversion["outputs"] != artifacts
        ):
            raise ContractError("no reviewed hash-bound conversion record")
        require_sha(conversion["sourceSha256"], "sourceSha256")
        require_sha(approved_verification_sha256, "approvedTensorEqualityVerificationSha256")
        if conversion["tensorEqualityVerificationSha256"] != approved_verification_sha256:
            raise ContractError("independent tensor equality verification identity mismatch")
        require_sha(conversion_sha256, "approvedConversionRecordSha256")
        if digest(conversion) != conversion_sha256:
            raise ContractError("conversion record identity mismatch")
    return weights
