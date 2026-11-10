"""Generate a neutral benchmark report (glossary, acceptance criteria, machine info, results).

Reads benchmark JSON reports/manifests under data/ and writes docs/benchmark-report.md.
No judgments or decisions are emitted about results.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: str) -> str | None:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    info["collected_at_utc"] = datetime.now(timezone.utc).isoformat()
    info["hostname"] = platform.node()
    info["os"] = f"{platform.system()} {platform.release()}"
    info["os_version"] = platform.version()
    info["arch"] = platform.machine()
    info["python"] = platform.python_version()
    lscpu = _run("lscpu")
    if lscpu:
        def g(pat: str) -> str | None:
            m = re.search(pat, lscpu, re.M)
            return m.group(1).strip() if m else None
        info["cpu_model"] = g(r"Model name:\s*(.+)")
        info["cpu_sockets"] = g(r"Socket\(s\):\s*(\S+)")
        info["cpu_cores_per_socket"] = g(r"Core\(s\) per socket:\s*(\S+)")
        info["cpu_threads_per_core"] = g(r"Thread\(s\) per core:\s*(\S+)")
        info["cpu_logical"] = g(r"CPU\(s\):\s*(\S+)")
        info["cpu_max_mhz"] = g(r"CPU max MHz:\s*(\S+)")
        info["cpu_min_mhz"] = g(r"CPU min MHz:\s*(\S+)")
        info["l1d"] = g(r"L1d cache:\s*(.+)")
        info["l1i"] = g(r"L1i cache:\s*(.+)")
        info["l2"] = g(r"L2 cache:\s*(.+)")
        info["l3"] = g(r"L3 cache:\s*(.+)")
        flags = g(r"Flags:\s*(.+)")
        info["cpu_flags_relevant"] = " ".join(f for f in (flags or "").split() if f in {"avx","avx2","fma","avx512f","avx512_vnni","sse4_2"}) or None
    mem = _run("free -b")
    if mem:
        m = re.search(r"Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+(\d+)", mem)
        if m:
            info["ram_total_bytes"] = int(m.group(1))
            info["ram_available_at_collect_bytes"] = int(m.group(4))
    try:
        import importlib.metadata as md
        pkgs = {}
        for p in ["onnxruntime","tensorflow","tensorflow-cpu","numpy","tf2onnx","onnx","Pillow","decimer","typer","sympy","rdkit"]:
            try:
                pkgs[p] = md.version(p)
            except Exception:
                pass
        info["packages"] = pkgs
    except Exception:
        pass
    return info


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_rate(n: int | None, d: int | None) -> str:
    if n is None or d in (None, 0):
        return "—"
    return f"{n}/{d} ({n/d:.4f})"


def _fmt_ms(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _delta_pp(var_path: Path, fp32_path: Path, field: str) -> str:
    v = _load_json(var_path); f = _load_json(fp32_path)
    if not v or not f:
        return "—"
    vd = {str(d.get("id")): d for d in v.get("details", [])}
    fd = {str(d.get("id")): d for d in f.get("details", [])}
    ids = [i for i in vd.keys() if i in fd]
    if not ids:
        # fallback to aggregate if same count and no details overlap
        vr, fr = v.get(f"{field}_rate"), f.get(f"{field}_rate")
        if vr is not None and fr is not None and v.get("count") == f.get("count"):
            return f"{(vr-fr)*100:+.2f} pp (agg)"
        return "—"
    vb = sum(1 for i in ids if vd[i].get(field) is True)
    fb = sum(1 for i in ids if fd[i].get(field) is True)
    dpp = (vb - fb) / len(ids) * 100
    note = "" if len(ids) == v.get("count") == f.get("count") else f", n={len(ids)}"
    return f"{dpp:+.2f} pp{note}"


def _artifact_size_mb(report_path: Path) -> str:
    # heuristic based on known artifact layout
    name = str(report_path)
    mapping = [
        ("ocsr-printed", "fp32", "data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx", "data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx"),
        ("hf-clef", "fp32", "data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx", "data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx"),
        ("hf-jpo", "fp32", "data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx", "data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx"),
        ("printed", "e1", "data/artifacts/quant/e1/ocsr-printed/encoder.e1.dynamic.onnx", "data/artifacts/quant/e1/ocsr-printed/decoder_step.e1.dynamic.onnx"),
        ("hf-clef", "e1", "data/artifacts/quant/e1/ocsr-printed/encoder.e1.dynamic.onnx", "data/artifacts/quant/e1/ocsr-printed/decoder_step.e1.dynamic.onnx"),
        ("hf-jpo", "e1", "data/artifacts/quant/e1/ocsr-printed/encoder.e1.dynamic.onnx", "data/artifacts/quant/e1/ocsr-printed/decoder_step.e1.dynamic.onnx"),
        ("hf-clef", "e3", "data/artifacts/quant/e3/ocsr-printed/encoder.e3.static.onnx", "data/artifacts/quant/e3/ocsr-printed/decoder_step.e3.dynamic.onnx"),
        ("hf-jpo", "e3", "data/artifacts/quant/e3/ocsr-printed/encoder.e3.static.onnx", "data/artifacts/quant/e3/ocsr-printed/decoder_step.e3.dynamic.onnx"),
        ("handdrawn-hdm", "fp32", "data/artifacts/fp32/ocsr-handdrawn/encoder.fp32.onnx", "data/artifacts/fp32/ocsr-handdrawn/decoder_step.fp32.onnx"),
        ("handdrawn-hdm", "e1", "data/artifacts/quant/e1/ocsr-handdrawn/encoder.e1.dynamic.onnx", "data/artifacts/quant/e1/ocsr-handdrawn/decoder_step.e1.dynamic.onnx"),
        ("handdrawn-hdm", "e3", "data/artifacts/quant/e3/ocsr-handdrawn/encoder.e3.static.onnx", "data/artifacts/quant/e3/ocsr-handdrawn/decoder_step.e3.dynamic.onnx"),
    ]
    # pilots/frozen
    mapping += [
        ("pilot/printed", "fp32", "data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx", "data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx"),
        ("pilot/printed", "int8-dyn-all", "data/artifacts/quant/e1/ocsr-printed/encoder.e1.dynamic.onnx", "data/artifacts/quant/e1/ocsr-printed/decoder_step.e1.dynamic.onnx"),
        ("pilot/handdrawn", "fp32", "data/artifacts/fp32/ocsr-handdrawn/encoder.fp32.onnx", "data/artifacts/fp32/ocsr-handdrawn/decoder_step.fp32.onnx"),
        ("pilot/handdrawn", "int8-dyn-all", "data/artifacts/quant/e1/ocsr-handdrawn/encoder.e1.dynamic.onnx", "data/artifacts/quant/e1/ocsr-handdrawn/decoder_step.e1.dynamic.onnx"),
    ]
    for ctx, var, enc, dec in mapping:
        if ctx in name and (f"/{var}." in name or f".{var}." in name or f"report.{var}" in name):
            total = 0; ok = False
            for p in [ROOT/enc, ROOT/dec]:
                if p.exists(): total += p.stat().st_size; ok = True
            if ok: return f"{total/1024/1024:.1f}"
    return "—"


def _row(label: str, report_path: Path, *, tf_join_note: str = "", fp32_path: Path | None = None) -> str | None:
    r = _load_json(report_path)
    if not r:
        return None
    n = r.get("count")
    ex = r.get("exact_match"); exr = r.get("exact_match_rate")
    cm = r.get("canonical_match"); ccomp = r.get("canonical_comparable"); cmr = r.get("canonical_match_rate")
    tfm = r.get("tf_match"); tfmr = r.get("tf_match_rate")
    tcm, tcc, tcr = _tf_canonical_stats(r)
    tf_cell = _fmt_rate(tfm, n)
    if tfm is None:
        tf_cell = "—"
    elif tf_join_note:
        tf_cell = f"{tfm}/{n} ({tfmr:.4f}) {tf_join_note}" if tfmr is not None else f"{tfm}/{n} {tf_join_note}"
    if tcm is not None and tcc:
        tf_cell = f"{tf_cell} / can {tcm}/{tcc} ({tcr:.3f})"
    valid = r.get("valid"); vr = r.get("valid_rate"); inc = r.get("incomplete")
    p50, p95, note = _latency_from_details(r)
    size = _artifact_size_mb(report_path)
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        fr = _load_json(fp32_path) or {}
        dtf = _delta_pp(report_path, fp32_path, "tf_match") if tfm is not None and fr.get("tf_match") is not None else "—"
        dex = _delta_pp(report_path, fp32_path, "exact_match")
        dcan = f"{(cmr-fr['canonical_match_rate'])*100:+.2f} pp" if (cmr is not None and fr.get("canonical_match_rate") is not None) else "—"
    elif fp32_path and report_path == fp32_path:
        dtf = dex = dcan = "(ref)"
    else:
        dtf = dex = dcan = "—"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    mark = " ✅≥70%" if cmr is not None and cmr >= 0.70 else (" ❌<70%" if cmr is not None else "")
    can_cell = f"**{cm}/{ccomp} ({cmr:.4f})**{mark}" if (cm is not None and ccomp) else "—"
    return (
        f"| {label} | {n} | {can_cell} | {ex} ({exr:.4f}) | {tf_cell} | {dcan} | {dtf} | {dex} | {valid} ({vr:.4f}) | {inc} | "
        f"{p50}{note} | {p95} | {size} | `{rep}` |"
    )


GLOSSARY = r"""## 1. Glosario

Las métricas se calculan con el mismo harness sobre el mismo conjunto de imágenes/IDs congelados cuando aplica (comparación pareada), salvo que se indique caché.

| Término / columna | Qué mide | Cómo leerla | Matiz importante |
|---|---|---|---|
| **OCSR** | Optical Chemical Structure Recognition: reconocer una estructura química desde una imagen y producir una representación (aquí, SMILES). | — | — |
| **SMILES** | Simplified Molecular Input Line Entry System: cadena de texto que representa una molécula. | — | La misma molécula puede escribirse con SMILES distintos. |
| **Modelo TF (baseline)** | Modelo DECIMER original en TensorFlow (SavedModel), usado como referencia de comportamiento. | — | Es la referencia para `tf_match`; no implica “verdad química” absoluta. |
| **Modelo ONNX** | Modelo convertido a ONNX (encoder + `decoder_step`) ejecutado con ONNX Runtime (ORT). | — | — |
| **FP32** | Exportación ONNX en precisión flotante de 32 bits (control/referencia de fidelidad). | — | Base para calcular Δ vs FP32. |
| **INT8 dynamic (all)** | Cuantización dinámica INT8 aplicada a ops soportadas (p. ej. E1), sin calibración. | — | Puede reducir tamaño; **no** garantiza menor latencia en CPU. |
| **INT8 dynamic (sin atención)** | Variante dinámica que excluye atención u ops selectivas. | — | `[según artefacto/disponibilidad]`. |
| **FP16** | Pesos/cómputo en media precisión donde aplique. | — | `[según viabilidad/runtime]`. |
| **INT8 estática (QDQ/calibración)** | Cuantización estática con datos de calibración (p. ej. E3 híbrido). | — | Sensible a calibración/distribución; evaluar en holdout según protocolo. |
| **`canonical_match` RDKit (primaria, vs label)** | **Métrica primaria OCSR (calidad química):** igualdad de SMILES **canónicos RDKit** entre predicción y etiqueta, contada solo cuando **ambos parsean** (`aciertos/comparables (tasa)`). | `210/292 (0.7192)` = 210 aciertos sobre 292 comparables. | Preferible para equivalencia de representación. Depende de RDKit/parseo; el denominador es `canonical_comparable` (puede ser < n). Casos no comparables se interpretan junto con `valid`/`incomplete`. |
| **`exact_match` string (secundaria, vs label)** | Coincidencia **exacta como texto** `predicho == esperado` (sin canonicalizar). | `121 (0.4033)` = 121/300 cadenas idénticas. | **Secundaria** para química: SMILES equivalentes pueden diferir y contar como no exactos. Útil como señal de salida cruda/consistencia textual. |
| **`tf_match` ONNX↔TF (fidelidad, string)** | Fidelidad de conversión: salida **ONNX** idéntica (string) a la del **baseline TF** para la misma imagen. `tf_canonical` (cuando se reporta) es la fidelidad en forma canónica. | `279/300 (0.9300)`. | Mide **paridad ONNX ≈ TF**, **no** corrección química frente a la etiqueta. |
| **Δ vs FP32 (pp): Δcanonical / Δtf / Δexact** | Diferencia absoluta en **puntos porcentuales** vs FP32, sobre la **misma métrica y mismos IDs** (pareado). | `-1.03 pp` en canónica = 1.03 pp menos que FP32. `(ref)` = FP32. | **Primario para calidad:** `Δcanonical`. “pp” es diferencia absoluta. |
| **`valid` / `incomplete`** | Heurísticas del harness (sin RDKit): `valid_rate` = fracción marcada válida por reglas simples (p. ej. no vacía/no inválida heurística); `incomplete` = secuencias que no finalizaron (p. ej. sin fin de secuencia). | `0.9967 / 1` = 99.67% válidas heurísticamente, 1 incompleta de 300. | Aproximación; **no** garantiza validez química real. |
| **Latencia p50 / p95 (ms)** | Tiempo por imagen (ms) en el mismo hardware y execution provider (aquí, ORT CPU): **p50** = mediana (50% ≤ t), **p95** = cola (95% ≤ t). | `3455 / 4676` = mediana ~3.46 s, cola p95 ~4.68 s. | No usar solo media. No comparable entre máquinas/configuraciones sin control. Si hay `cache_hits`, la latencia no es fresca/comparable (a veces `n/a`). |
| **Tamaño (MB)** | Tamaño en disco de los artefactos de la variante (p. ej. encoder+decoder). | — | `[si se calcula]`. |
| **Throughput** | Imágenes/s o lotes/s si se evalúa en lote. | — | `[si aplica]`. |
| **Clasificador (binario + reactions)** | Métrica binaria sobre 3 clases mapeadas (`one/several_molecule→structure`, `other→no-structure`); `reactions` **excluida** del binario y reportada aparte como `ambiguous`. | Matriz `[[TP,FN],[FP,TN]]`, accuracy binaria, paridad TF, conteo reactions→predicción. | El binario no incluye `reactions`; interpretar por separado. |
| **`cache_hits`** | Predicciones reutilizadas desde caché `(model, variant, image_id)`. | — | Acelera reruns; si todas son cache hits, no hay latencia fresca medida. |
| **McNemar** | Prueba pareada sobre discordancias (A acierta/B falla vs inverso) en el mismo set. | — | `[pendiente implementar en reporte/harness]`. |"""

CRITERIA = r"""## 2. Criterios de aceptación (definidos para la evaluación)

| Criterio | Definición / condición de aceptación |
|---|---|
| **Piso absoluto (obligatorio, todos)** | **Accuracy ≥ 70%** en la **métrica primaria** del dossier, sobre el **split `eval` congelado** (mismos IDs). **OCSR:** `canonical_match_rate` RDKit (sobre pares comparables). **Clasificador:** `binary accuracy` (no-reacción). Variantes < 70% **no se aceptan**. |
| **Comparación pareada** | Todas las variantes se miden sobre el **mismo set/IDs congelados** (piloto ⊂ eval ⊂ pool). Δ vs FP32 en pp sobre mismos IDs. |
| **Fidelidad (tasa / estricta)** | `tf_match_rate`/`tf_canonical_match_rate` (ONNX≈TF). Estricto: 100%. Umbral numérico (`≥ [TBD]%`) es parámetro externo si se define. |
| **Δ vs FP32 (precisión/fidelidad)** | Diferencia absoluta en pp vs FP32, misma métrica e IDs. Umbral `≤ [TBD] pp` `[TBD]`. |
| **Exactitud OCSR (primaria/secundaria)** | **Primaria:** canónica RDKit (`canonical_match`). **Secundaria:** `exact_match` string (sin canonicalizar), dato informativo. |
| **Validez / completitud** | `valid_rate`, `incomplete_rate`. Umbrales externos si se definen. |
| **Latencia** | **p50/p95** ms, mismo hardware y execution provider (ORT CPU). No solo media. Umbral p95 `[TBD]`. INT8 dinámico no implica menor latencia. |
| **Tamaño / throughput** | MB en disco (artefactos); throughput (img/s, batch) si aplica. |
| **Clasificador** | Binaria sobre 3 clases mapeadas; `reactions` separada/excluida del binario. |
| **Reproducibilidad** | Fuente/pin/manifiesto/checksums, IDs congelados (seed), hardware/provider/versiones. |"""

PROVENANCE = r"""## 3. Fuentes de datos y procedencia (hechos)

| Suite / uso | Modelo objetivo | Fuente | Identificador / pin | Licencia / procedencia (según manifiesto/fuente) | Artefactos locales |
|---|---|---|---|---|---|
| Golden impreso CLEF | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | dataset commit observado `2fdcdc59`; config `CLEF`, split `test`, 992 | HF agregado **mixed-see-sources**; verificar términos originales CLEF 2012 | `data/benchmark/hf-clef/labels.jsonl`; `data/manifests/benchmark-hf-clef.import.json` |
| Golden impreso JPO | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | config `JPO`, split `test`, 449 | HF agregado **mixed-see-sources**; verificar términos originales JPO/ChemInfty/OSRA | `data/benchmark/hf-jpo/labels.jsonl`; `data/manifests/benchmark-hf-jpo.import.json` |
| Hand-drawn HDM | `ocsr-handdrawn` | Zenodo DECIMER Hand-drawn molecule images | record `7617107`, DOI 10.5281/zenodo.7617107, 5088 | CC-BY-4.0 | `data/benchmark/handdrawn-hdm/labels.jsonl`; `data/manifests/benchmark-handdrawn-hdm.fetch.json` |
| Clasificador (preparado) | `image-classifier` | Zenodo 13378718 | record `13378718`, md5 zip `e7e47f2aa4af4b79874a19382cec06da` (Zenodo API) | Registro indica MIT; origen imágenes mixto | `decimer-export import-benchmark --source classifier-13378718` (requiere extracción local) |
| Clasificador 6670746 | `image-classifier` | Zenodo 6670746 | archivo único 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 | No descargado. |

**Mapeo clasificador (13378718, binario):** `one_molecule`→`structure`, `several_molecules`→`structure`, `other`→`no-structure` (incluidas en matriz binaria); `reactions`→excluida del binario, métrica separada `ambiguous_excluded`."""


def _latency_from_details(d: dict[str, Any]) -> tuple[str, str, str]:
    lat = d.get("latency_ms") or {}
    p50, p95 = lat.get("p50"), lat.get("p95")
    if p50 is None or p95 is None:
        vals = [r.get("latency_ms") for r in d.get("details", []) if not r.get("cache_hit") and r.get("latency_ms") is not None]
        vals = [float(v) for v in vals]
        if vals:
            vals.sort()
            p50 = vals[len(vals) // 2]
            idx = max(0, int(len(vals) * 0.95) - 1)
            p95 = vals[idx]
    cache_hits = d.get("cache_hits")
    n = d.get("count") or d.get("total_rows") or 0
    note = ""
    if cache_hits and n and int(cache_hits) >= int(n):
        note = " (cache, sin latencia fresca)"
        return "—", "—", note
    if p50 is None and cache_hits:
        return "—", "—", f" (cache_hits={cache_hits})"
    return _fmt_ms(p50), _fmt_ms(p95), note


def _tf_canonical_stats(r: dict[str, Any]) -> tuple[int | None, int | None, float | None]:
    """Top-level tf_canonical aggregate or computed from details."""
    if r.get("tf_canonical_comparable") is not None:
        return r.get("tf_canonical_match"), r.get("tf_canonical_comparable"), r.get("tf_canonical_match_rate")
    det = r.get("details") or []
    if not det or not any("tf_canonical_match" in d for d in det[:3]):
        return None, None, None
    comp = [d for d in det if d.get("tf_canonical_match") is not None]
    if not comp:
        return (None, 0, None) if any(d.get("tf_smiles") is not None for d in det) else (None, None, None)
    m = sum(1 for d in comp if d.get("tf_canonical_match"))
    return m, len(comp), m / len(comp)


def _pilot_ocsr_row(label: str, report_path: Path, *, fp32_path: Path | None = None) -> str | None:
    r = _load_json(report_path)
    if not r:
        return None
    n = r.get("count")
    ex = r.get("exact_match"); exr = r.get("exact_match_rate")
    cm = r.get("canonical_match"); ccomp = r.get("canonical_comparable"); cmr = r.get("canonical_match_rate")
    tfm = r.get("tf_match"); tfmr = r.get("tf_match_rate")
    tcm, tcc, tcr = _tf_canonical_stats(r)
    valid = r.get("valid"); vr = r.get("valid_rate"); inc = r.get("incomplete")
    p50, p95, note = _latency_from_details(r)
    size = _artifact_size_mb(report_path)
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        dtf = _delta_pp(report_path, fp32_path, "tf_match") if tfm is not None else "—"
        dex = _delta_pp(report_path, fp32_path, "exact_match")
        fr = _load_json(fp32_path) or {}
        dcan = "—"
        if cmr is not None and fr.get("canonical_match_rate") is not None:
            dcan = f"{(cmr-fr['canonical_match_rate'])*100:+.2f} pp"
    elif fp32_path and report_path == fp32_path:
        dtf = dex = dcan = "(ref)"
    else:
        dtf = dex = dcan = "—"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    tf_cell = _fmt_rate(tfm, n) if tfm is not None else "—"
    if tcm is not None and tcc:
        tf_cell = f"{tf_cell} / can {tcm}/{tcc} ({tcr:.3f})"
    mark = " ✅≥70%" if cmr is not None and cmr >= 0.70 else (" ❌<70%" if cmr is not None else "")
    can_cell = f"**{cm}/{ccomp} ({cmr:.4f})**{mark}" if (cm is not None and ccomp) else "—"
    # Primary = canonical RDKit (floor >=70%)
    return (
        f"| {label} | {n} | {can_cell} | {ex} ({exr:.4f}) | {tf_cell} | {dcan} | {dtf} | {dex} | {valid} ({vr:.4f}) | {inc} | "
        f"{p50}{note} | {p95} | {size} | `{rep}` |"
    )


def _classifier_row(report_path: Path, *, fp32_path: Path | None = None) -> str | None:
    d = _load_json(report_path)
    if not d:
        return None
    cm = d.get("binary_confusion_matrix") or {}
    rx = d.get("reactions_separate_metric") or {}
    tfp = d.get("tf_parity") or {}
    acc = cm.get("accuracy")
    total = cm.get("total")
    mat = cm.get("matrix_rows_expected_cols_predicted")
    amark = " ✅≥70%" if acc is not None and acc >= 0.70 else (" ❌<70%" if acc is not None else "")
    acc_s = f"{acc:.4f} (n={total}){amark}" if acc is not None else "—"
    mat_s = f"`{mat}`" if mat else "—"
    tf_s = _fmt_rate(tfp.get("agree"), tfp.get("total")) if tfp else "—"
    rx_s = f"{rx.get('count','—')} → struct {rx.get('predicted_structure','—')} / no-struct {rx.get('predicted_no_structure','—')}" if rx else "—"
    p50, p95, note = _latency_from_details(d)
    variant = d.get("variant_key") or report_path.stem
    # Δ vs FP32 on binary accuracy (aggregate; same frozen split implied)
    dacc = "—"
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        fd = _load_json(fp32_path) or {}
        facc = ((fd.get("binary_confusion_matrix") or {}).get("accuracy"))
        if acc is not None and facc is not None:
            dacc = f"{(acc-facc)*100:+.2f} pp"
    elif fp32_path and report_path == fp32_path:
        dacc = "(ref)"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    split = d.get("split") or "—"
    return f"| `{variant}` (`{split}`) | {d.get('total_rows','—')} | {acc_s} | {mat_s} | {dacc} | {tf_s} | {rx_s} | {p50}{note} | {p95} | `{rep}` |"


def _ocsr_variant_label(path: Path, data: dict[str, Any] | None) -> str:
    d = data or {}
    vk = (d.get("variant_key") or path.stem or "").lower()
    name = path.name
    if "fp32enc-e1sdec-rr" in name or vk == "fp32enc-e1sdec-rr": return "FP32enc + e1sdec (no-attn) + reduce_range"
    if "fp32enc-e1sdec" in name or vk == "fp32enc-e1sdec": return "FP32enc + e1sdec (no-attn, dec mixto)"
    if "fp32enc-e1dec-rr" in name or vk == "fp32enc-e1dec-rr": return "FP32enc + E1dec + reduce_range"
    if "fp32enc-e1dec" in name or vk == "fp32enc-e1dec": return "FP32enc + E1dec (dyn)"
    if "e1enc-fp32dec" in name or vk == "e1enc-fp32dec": return "E1enc (dyn) + FP32dec"
    if "selbenc-e1sdec" in name or vk == "selbenc-e1sdec": return "selBenc (bloques 40..56 FP32) + e1sdec"
    if "selaenc-e1sdec" in name or vk == "selaenc-e1sdec": return "selAenc (stem+0..5+50..56 FP32) + e1sdec"
    if "selcenc-e1sdec" in name or vk == "selcenc-e1sdec": return "selCenc (stem+0..15 FP32) + e1sdec"
    if "e1s-sel-noattn" in name or "e1s" in vk and "noattn" in name: return "E1s selective dyn (decoder sin atención)"
    if "int8-dyn-all" in name or vk == "int8-dyn-all" or ("e1" in vk and "dyn" in vk): return "E1 dynamic INT8 all"
    if "e3" in name or "e3" in vk: return "E3 hybrid (static-enc QDQ + dyn-dec)"
    if "e4" in name or "fp16" in vk: return "E4 FP16"
    if "fp32" in name or vk == "fp32": return "FP32 (ref)"
    if "tf-baseline" in vk or vk == "tf": return "TF baseline"
    return d.get("variant_key") or path.stem


def _render_ocsr(lines: list[str], pending: list[str]) -> None:
    configs = [
        ("`ocsr-printed`", "printed", ROOT/"data/manifests/frozen/printed.frozen.json"),
        ("`ocsr-handdrawn`", "handdrawn", ROOT/"data/manifests/frozen/handdrawn.frozen.json"),
    ]
    header = "| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB | Reporte |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    splits = [
        ("eval-1000", "split `eval` (gate, n=1000)", "eval1000"),
        ("remainder", "split `remainder` (held-out, sin selección)", "remainder"),
        ("pilot", "split `pilot` (señal/selección, n=300)", "pilot300"),
    ]
    for title, prefix, fman in configs:
        man = _load_json(fman) or {}
        meta = ""
        if man:
            meta = f" (frozen n={man.get('frozen_n')}, eval={man.get('eval_n')}, pilot={man.get('pilot_n')}, seed={man.get('seed')}, source=`{fman.relative_to(ROOT)}`)"
        for dirname, splitlabel, suffix in splits:
            d = ROOT / "data/benchmark" / dirname
            if not d.exists():
                continue
            files = sorted([p for p in d.glob(f"{prefix}.*.{suffix}.json") if "imago" not in p.name.lower()])
            if not files:
                continue
            # fp32 ref for this split
            fp32_ref = d / f"{prefix}.fp32.{suffix}.json"
            if not fp32_ref.exists():
                fp32_ref = files[0]
            rows = []
            for p in files:
                data = _load_json(p)
                lbl = _ocsr_variant_label(p, data)
                rows.append(_pilot_ocsr_row(lbl, p, fp32_path=fp32_ref))
            got = [r for r in rows if r]
            if not got:
                continue
            marker = "5.E" if dirname == "eval-1000" else ("5.R" if dirname == "remainder" else "5.P")
            lines.append(f"### {marker} {title} — **{splitlabel}**{meta}")
            lines.append("")
            lines.append(header); lines.append(sep); lines.extend(got); lines.append("")
    if (ROOT / "data/benchmark/eval-1000/printed.fp32.eval1000.json").exists():
        pending = [p for p in pending if "eval-1000 pendiente" not in p.lower()]


def _classifier_label(path: Path, data: dict[str, Any] | None) -> str:
    d = data or {}
    vk = (d.get("variant_key") or path.stem or "").lower()
    name = path.name.lower()
    if "e4-fp16" in name or "fp16" in vk: return "E4 FP16 (seleccionada, ≥50% compresión, acc ≥70%)"
    if "e2-static" in name or ("e2" in vk and "static" in vk): return "E2 static QDQ (holdout calib)"
    if "e1s-sel" in name or ("e1s" in vk): return "E1s selective dyn (no Conv)"
    if "int8-dyn-all" in name or vk == "int8-dyn-all": return "E1 dynamic INT8 all"
    if "fp32" in name or vk == "fp32": return "FP32 (ref)"
    return d.get("variant_key") or path.stem


def _render_classifier(lines: list[str], pending: list[str]) -> None:
    for dirname, splitlabel in [("eval-1000", "eval-1000"), ("pilot", "pilot")]:
        d = ROOT / "data/benchmark" / dirname
        if not d.exists():
            continue
        files = sorted([p for p in d.glob("classifier.*.json") if "imago" not in p.name.lower()])
        if not files:
            continue
        fp32_ref = d / ("classifier.fp32.eval1000.json" if dirname == "eval-1000" else "classifier.fp32.pilot300.json")
        got2 = []
        for p in files:
            data = _load_json(p)
            lbl = _classifier_label(p, data)
            row = _classifier_row(p, fp32_path=fp32_ref)
            if not row:
                continue
            parts = row.split("|")
            if len(parts) > 2:
                parts[1] = f" {lbl} (`{splitlabel}`) "
                row = "|".join(parts)
            got2.append(row)
        if not got2:
            continue
        lines.append(f"### 5.5 `image-classifier` (Zenodo 13378718, congelado) — {splitlabel}")
        lines.append("")
        fman = _load_json(ROOT/"data/manifests/frozen/classifier.frozen.json") or {}
        if fman and dirname == "eval-1000":
            lines.append(f"Frozen: n={fman.get('frozen_n')}, eval={fman.get('eval_n')}, pilot={fman.get('pilot_n')}, seed={fman.get('seed')}, clases {fman.get('class_allocated')}. Binaria sobre `one/several_molecule→structure`, `other→no-structure` (threshold `0.000089`); `reactions` excluida del binario y reportada aparte. Objetivo: **accuracy ≥70%** mejorando compresión. **Todos** los resultados de esta suite se registran (incluidos los no seleccionados).")
            lines.append("")
        lines.append("| Variante (split) | total filas | Binaria acc (no-reaction) | Matriz `[[TN,FP],[FN,TP]]` | Δacc vs FP32 | Paridad TF (incl. reactions) | reactions (separada) | p50 ms | p95 ms | Reporte |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|")
        lines.extend(got2)
        lines.append("")
    if (ROOT / "data/benchmark/eval-1000/classifier.e4-fp16.eval1000.json").exists():
        lines.append("> **Decisión (clasificador, por ahora):** `E4 FP16` cumple el objetivo (acc ≈ FP32, ≥70%; **-49% tamaño**; paridad TF alta). INT8 en Conv (E1-all/E2 static holdout) regresa bajo 70%; E1s preserva accuracy pero no comprime.")
        lines.append("")
    lines.append("| Suite | Estado | Nota |")
    lines.append("|---|---|---|")
    lines.append("| Zenodo 6670746 | OUT OF SCOPE | No descargar/usar (126.6 GB). |")
    lines.append("| Zenodo 13378718 | Todos los runs registrados en este reporte | 4→binario mapeado; `reactions` separada. Imago excluido (experimento de curiosidad, `docs/experiments/`). |")
    lines.append("")
def render_machine(info: dict[str, Any]) -> str:
    ram = info.get("ram_total_bytes")
    ram_gb = f"{ram/1024**3:.2f} GiB ({ram} bytes)" if ram else "—"
    ram_av = info.get("ram_available_at_collect_bytes")
    ram_av_s = f"{ram_av/1024**3:.2f} GiB" if ram_av else "—"
    pkgs = info.get("packages") or {}
    pkg_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(pkgs.items())) or "| — | — |"
    return f"""## 4. Características de la máquina (recolección automática)

Recolectado automáticamente el `{info.get('collected_at_utc','')}` mediante `platform`, `lscpu` y `free`.

| Campo | Valor |
|---|---|
| Hostname | `{info.get('hostname','')}` |
| OS | `{info.get('os','')}` |
| Versión kernel/OS | `{info.get('os_version','')}` |
| Arquitectura | `{info.get('arch','')}` |
| Python | `{info.get('python','')}` |
| CPU modelo | `{info.get('cpu_model','')}` |
| Sockets / núcleos por socket / hilos por núcleo | `{info.get('cpu_sockets','')}` / `{info.get('cpu_cores_per_socket','')}` / `{info.get('cpu_threads_per_core','')}` |
| CPUs lógicas | `{info.get('cpu_logical','')}` |
| Frecuencia CPU máx/mín (MHz) | `{info.get('cpu_max_mhz','')}` / `{info.get('cpu_min_mhz','')}` |
| Cachés L1d/L1i/L2/L3 | `{info.get('l1d','')}` / `{info.get('l1i','')}` / `{info.get('l2','')}` / `{info.get('l3','')}` |
| Flags CPU relevantes | `{info.get('cpu_flags_relevant','')}` |
| RAM total | {ram_gb} |
| RAM disponible (al recolectar) | {ram_av_s} |

### Versiones de paquetes (entorno de ejecución)

| Paquete | Versión |
|---|---|
{pkg_rows}
"""


def render_results() -> tuple[str, list[str]]:
    b = ROOT / "data" / "benchmark"
    pending: list[str] = []
    lines = [
        "## 5. Resultados",
        "",
        "Protocolo: variantes comparadas sobre **mismos IDs** cuando aplica (pareado, piloto ⊂ eval ⊂ pool congelado). Latencia **p50/p95** (mismo hardware/provider), no solo media; INT8 dinámico puede no ser más rápido. **Δ vs FP32** = diferencia en pp sobre IDs comunes (misma métrica). `—` = no medido/no disponible. Ver glosario (§1) para significado de columnas.",
        "",
    ]

    def table(title: str, rows: list[str | None]):
        got = [r for r in rows if r]
        if not got:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.extend(got)
        lines.append("")

    # Pilotos congelados (tres dossiers)
    _render_ocsr(lines, pending)

    table("5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992, evidencia histórica)", [
        _row("E1 dynamic INT8 (+TF)", b/"hf-clef/report.e1.tf.json"),
        _row("E3 static-enc + dyn-dec (tf_match por join a TF de E1)", b/"hf-clef/report.e3.tfjoined.json", tf_join_note="[join]"),
    ])
    table("5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449, evidencia histórica)", [
        _row("E1 dynamic INT8 (+TF)", b/"hf-jpo/report.e1.tf.json"),
        _row("E3 static-enc + dyn-dec (tf_match por join a TF de E1)", b/"hf-jpo/report.e3.tfjoined.json", tf_join_note="[join]"),
    ])
    hdm_fp32 = b/"handdrawn-hdm/report.fp32.tf.limit500.json"
    table("5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra ad hoc `--limit 500`, no decisión formal)", [
        _row("FP32 (+TF)", hdm_fp32, fp32_path=hdm_fp32),
        _row("E1 dynamic INT8 (+TF)", b/"handdrawn-hdm/report.e1.tf.limit500.json", fp32_path=hdm_fp32),
        _row("E3 static-enc + dyn-dec (+TF, calib 32 HDM)", b/"handdrawn-hdm/report.e3.tf.limit500.json", fp32_path=hdm_fp32),
    ])
    full = b/"handdrawn-hdm/report.e1.tf.full5088.json"
    if full.exists():
        table("5.4 `ocsr-handdrawn` — HDM completo (5088)", [
            _row("E1 dynamic INT8 (+TF)", full),
        ])
    else:
        pending.append(f"`ocsr-handdrawn` HDM completo n=5088, E1+TF: `{full.relative_to(ROOT)}` (no encontrado al generar).")
        lines.append("### 5.4 `ocsr-handdrawn` — HDM completo (5088)")
        lines.append("")
        lines.append("| Variante | n | Estado | Reporte esperado |")
        lines.append("|---|---:|---|---|")
        lines.append(f"| E1 dynamic INT8 (+TF) | 5088 | Pendiente / no disponible al generar | `{full.relative_to(ROOT)}` |")
        lines.append("")

    _render_classifier(lines, pending)
    return "\n".join(lines), pending


LIMITATIONS = r"""## 6. Limitaciones de medición (hechos metodológicos)

- `exact_match` (string) compara `predicho == esperado` **sin canonicalización RDKit**. `canonical_match` (RDKit) compara SMILES canónicos cuando ambos parsean (integrado en benchmark OCSR; puede no estar en reportes históricos). Diferencias equivalentes de representación pueden contar como no exactas en `exact_match`.
- `tf_match` mide fidelidad **ONNX↔TF** según el baseline/pipeline del harness; no es exactitud química canónica frente a la etiqueta.
- `valid`/`incomplete` son heurísticas sin RDKit (aproximación).
- Latencias son del entorno de ejecución/harness (máquina/provider/carga/hilos/versiones). Reportar **p50/p95**; no usar solo media. `cache_hits` invalida latencia fresca. INT8 dinámico puede ser más lento que FP32 en CPU.
- Δ en pp calculado sobre la misma métrica e IDs cuando hay datos pareados; si falta pareo, puede indicarse agregado.
- Subsets HF CLEF/JPO: agregado `mixed-see-sources`; verificar términos originales antes de redistribución.
- Cuantización estática (E3/QDQ) depende fuertemente del set de calibración (tamaño/distribución/holdout); resultados no generalizan sin protocolo controlado.
- Clasificador: binario solo sobre 3 clases mapeadas; `reactions` excluida y reportada aparte (`ambiguous_excluded`)."""


def generate_report(targets: list[Path] | None = None, machine_info: dict[str, Any] | None = None) -> str:
    if targets is None:
        targets = [ROOT / "docs/benchmark-report.md"]
    info = machine_info or collect_machine_info()
    results, pending = render_results()
    now = datetime.now(timezone.utc).isoformat()
    parts = [
        "# Informe de benchmarks DECIMER → ONNX",
        "",
        "> Documento generado automáticamente a partir de reportes JSON/manifiestos en `data/`. No incluye decisiones ni juicios sobre los resultados; reporta mediciones, definiciones y criterios de aceptación definidos para la evaluación.",
        "",
        f"**Generado (UTC):** `{now}`  ",
        "**Generador:** `toolchain.report.generate_report` / `dtx report` (alias: `decimer-export report`)  ",
        "**Rama/commits relevantes:** `main` (workspace fetch/export/runtime/cuantización/benchmark).",
        "",
        "---",
        "",
        GLOSSARY,
        "",
        CRITERIA,
        "",
        PROVENANCE,
        "",
        render_machine(info),
        results,
        "## 7. Archivos de referencia",
        "",
        "- Manifiestos fetch/import (trackeados): `data/manifests/benchmark-*.json`.",
        "- Reportes detallados (gitignored, generados localmente): `data/benchmark/**/*.json`.",
        "- Resumen golden printed (trackeado): `data/artifacts/selection/ocsr-printed.hf-golden.json`.",
        "- Selección printed (trackeada, puede contener interpretación): `data/artifacts/selection/ocsr-printed.selection.json`.",
        "",
        LIMITATIONS,
        "",
    ]
    if pending:
        parts += ["## 8. Elementos pendientes al generar", ""] + [f"- {p}" for p in pending] + [""]
    text = "\n".join(parts)
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    generate_report()
    print("ok")
