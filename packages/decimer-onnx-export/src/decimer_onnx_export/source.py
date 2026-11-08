"""Pipeline stage 2 — Probe/extract (M0/M1).

Inspects fetched DECIMER assets (SavedModel, tokenizer, checkpoints) and reports what is
available for export. This is the decision point for ADR-0001/0002:
- extract encoder/decoder subgraphs from the published SavedModel, or
- rebuild encoder/decoder from source config and restore checkpoints.

TensorFlow is imported lazily so non-probe commands stay light where possible.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data/weights")


@dataclass
class ProbeResult:
    model: str
    data_dir: str
    found: bool = False
    saved_models: list[str] = field(default_factory=list)
    tokenizers: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    signatures: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _find(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob(pattern) if ".part" not in p.name)


def _summarize_signature(obj: Any) -> Any:
    """Best-effort summary of TF concrete function signatures."""
    try:
        sig = getattr(obj, "signatures", None)
        if sig:
            out = {}
            for name, fn in sig.items():
                try:
                    out[name] = {
                        "inputs": {k: str(v) for k, v in fn.structured_input_signature[1].items()},
                        "outputs": str(fn.structured_outputs),
                    }
                except Exception as e:  # noqa: BLE001
                    out[name] = f"<unavailable: {e}>"
            return out
        if callable(obj):
            # tf.function / concrete function without named signatures
            try:
                cfs = obj._list_all_concrete_functions_for_serialization()  # noqa: SLF001
                return [str(cf.structured_input_signature) for cf in cfs[:4]]
            except Exception:
                return str(type(obj))
        return {"attributes": [a for a in dir(obj) if not a.startswith("_")][:40]}
    except Exception as e:  # noqa: BLE001
        return f"<signature error: {e}>"


def _inspect_saved_model(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        import tensorflow as tf  # local import: heavy
    except Exception as e:  # noqa: BLE001
        return {}, [f"tensorflow import failed: {e}"]
    try:
        loaded = tf.saved_model.load(str(path))
    except Exception as e:  # noqa: BLE001
        return {}, [f"failed to load {path}: {e}"]
    info: dict[str, Any] = {"path": str(path), "type": type(loaded).__name__}
    info["signatures"] = _summarize_signature(loaded)
    for attr in ("DECIMER", "encoder", "transformer", "tokenizer"):
        if hasattr(loaded, attr):
            info[f"has_{attr}"] = True
            info[f"{attr}_type"] = type(getattr(loaded, attr)).__name__
    return info, errors


def _inspect_tokenizer(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "size": path.stat().st_size}
    try:
        with path.open("rb") as f:
            tok = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
        return info
    index_word = getattr(tok, "index_word", None) or getattr(tok, "index_to_word", None)
    word_index = getattr(tok, "word_index", None)
    if word_index is not None:
        info["vocab_size"] = len(word_index)
    if index_word is not None:
        info["index_word_size"] = len(index_word)
        sample = list(index_word.items())[:8] if isinstance(index_word, dict) else index_word[:8]
        info["sample"] = sample
    return info


def probe_model(model: str, data_dir: Path = DEFAULT_DATA_DIR) -> ProbeResult:
    root = data_dir / model
    res = ProbeResult(model=model, data_dir=str(data_dir))
    if not root.exists():
        res.notes.append(f"not found: {root}. Run: uv run decimer-export fetch --model {model}")
        return res
    res.found = True

    saved = [p.parent for p in _find(root, "saved_model.pb")]
    # de-dup nested matches preferring top-level model dirs
    seen = set()
    for p in saved:
        rp = str(p)
        if rp not in seen:
            seen.add(rp)
            res.saved_models.append(rp)

    res.tokenizers = [str(p) for p in _find(root, "tokenizer*.pkl")]
    # checkpoints: TF checkpoint index or checkpoint dir
    res.checkpoints = [str(p) for p in _find(root, "checkpoint")]
    ckpt_idx = [str(p) for p in _find(root, "*.index") if "variables.index" not in p.name]
    for c in ckpt_idx:
        if c not in res.checkpoints:
            res.checkpoints.append(c)

    for sm in res.saved_models[:2]:
        info, errs = _inspect_saved_model(Path(sm))
        res.signatures[sm] = info
        res.errors.extend(errs)
    for tk in res.tokenizers[:2]:
        res.signatures[tk] = _inspect_tokenizer(Path(tk))

    if not res.saved_models and not res.tokenizers and not res.checkpoints:
        res.notes.append("no SavedModel/tokenizer/checkpoints discovered under model dir")
    if res.saved_models:
        res.notes.append("M0 next: confirm end-to-end TF parity, then export encoder+decoder-step to FP32 ONNX")
    return res


def format_probe(res: ProbeResult) -> str:
    lines = [f"probe: {res.model} (data_dir={res.data_dir})", f"  found: {res.found}"]
    for label, items in [
        ("saved_models", res.saved_models),
        ("tokenizers", res.tokenizers),
        ("checkpoints", res.checkpoints),
    ]:
        lines.append(f"  {label}: {len(items)}")
        for item in items:
            lines.append(f"    - {item}")
    for k, v in res.signatures.items():
        lines.append(f"  detail[{Path(k).name}]:")
        try:
            blob = json.dumps(v, indent=2, default=str)
        except Exception:
            blob = str(v)
        for bl in blob.splitlines()[:120]:
            lines.append(f"    {bl}")
    for n in res.notes:
        lines.append(f"  note: {n}")
    for e in res.errors:
        lines.append(f"  error: {e}")
    return "\n".join(lines)
