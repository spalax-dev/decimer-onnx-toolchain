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

ROOT = Path(__file__).resolve().parents[4]


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
        for p in ["onnxruntime","tensorflow","tensorflow-cpu","numpy","tf2onnx","onnx","Pillow","decimer","typer","sympy"]:
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
    tfm = r.get("tf_match"); tfmr = r.get("tf_match_rate")
    tf_cell = _fmt_rate(tfm, n)
    if tfm is None:
        tf_cell = "—"
    elif tf_join_note:
        tf_cell = f"{tfm}/{n} ({tfmr:.4f}) {tf_join_note}" if tfmr is not None else f"{tfm}/{n} {tf_join_note}"
    valid = r.get("valid"); vr = r.get("valid_rate"); inc = r.get("incomplete")
    lat = r.get("latency_ms") or {}
    size = _artifact_size_mb(report_path)
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        dtf = _delta_pp(report_path, fp32_path, "tf_match") if tfm is not None and (_load_json(fp32_path) or {}).get("tf_match") is not None else "—"
        dex = _delta_pp(report_path, fp32_path, "exact_match")
    elif fp32_path and report_path == fp32_path:
        dtf = dex = "(ref)"
    else:
        dtf = dex = "—"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    return (
        f"| {label} | {n} | {ex} ({exr:.4f}) | {tf_cell} | {dtf} | {dex} | {valid} ({vr:.4f}) | {inc} | "
        f"{_fmt_ms(lat.get('p50'))} | {_fmt_ms(lat.get('p95'))} | {size} | `{rep}` |"
    )


GLOSSARY = r"""## 1. Glosario

| Término | Definición |
|---|---|
| **OCSR** | Optical Chemical Structure Recognition: reconocimiento de estructuras químicas desde imágenes para producir una representación (aquí, SMILES). |
| **SMILES** | Simplified Molecular Input Line Entry System: cadena textual que representa una estructura molecular. |
| **Modelo TF (baseline)** | Modelo DECIMER original ejecutado en TensorFlow (SavedModel), usado como referencia de comportamiento. |
| **Modelo ONNX** | Modelo convertido a ONNX (encoder + `decoder_step`) ejecutado con ONNX Runtime (ORT). |
| **FP32** | Exportación ONNX en precisión flotante de 32 bits (control de fidelidad). |
| **INT8 dynamic (all)** | Cuantización dinámica INT8 aplicada a ops soportados (p. ej. E1); sin calibración. |
| **INT8 dynamic (sin atención)** | Variante dinámica que excluye atención (u ops selectivas) `[según artefacto]`. |
| **FP16** | Pesos/compute en media precisión donde aplique. |
| **INT8 estática (QDQ/calibración)** | Cuantización estática con calibración (p. ej. E3 híbrido); evaluación en holdout según protocolo. |
| **Δ vs FP32 (pp)** | Diferencia absoluta en puntos porcentuales respecto a FP32, calculada sobre la **misma métrica y los mismos IDs** (pareado): `métrica_variante − métrica_FP32`. |
| **`tf_match`** | Proporción de casos donde la salida ONNX es idéntica al baseline TF (fidelidad ONNX↔TF, según harness/protocolo). |
| **`exact_match` (vs etiqueta)** | Proporción de casos donde el SMILES predicho es idéntico como cadena a la etiqueta (`got == expected`); sensible a canonicalización. |
| **Exactitud canónica (RDKit)** | Comparación sobre SMILES canónico `[pendiente integrar en harness]`. |
| **`valid` / `incomplete`** | Heurísticas del harness (sin RDKit): no vacío/no incompleto; secuencia no finalizada. |
| **Latencia p50 / p95 (ms)** | Percentiles 50/95 por muestra (mismo hardware/provider). No usar solo media; p95 refleja cola. |
| **Tamaño (MB)** | Tamaño en disco de artefactos (encoder+decoder) de la variante. |
| **Throughput** | Imágenes/s o lotes/s si se evalúa en lote `[si aplica]`. |
| **McNemar** | Prueba pareada sobre pares discordantes (A acierta/B falla vs inverso) para comparar variantes en el mismo set. |"""

CRITERIA = r"""## 2. Criterios de aceptación (definidos para la evaluación)

Criterios definidos como referencia de medición (sin aplicar veredictos en este documento):

| Criterio | Definición / condición de aceptación |
|---|---|
| **Comparación pareada** | Todas las variantes se miden sobre el **mismo set/IDs congelados** (piloto ⊂ eval ⊂ pool). Δ vs FP32 en pp sobre mismos IDs. |
| **Fidelidad (tasa / estricta)** | `tf_match_rate = tf_match/count` (ONNX==TF). Estricto: 100%. Umbral numérico (`≥ [TBD]%`) es parámetro externo si se define. |
| **Δ vs FP32 (precisión/fidelidad)** | Diferencia absoluta en pp vs FP32, misma métrica e IDs. Umbral `≤ [TBD] pp` es parámetro externo si se define. |
| **Exactitud (canónica / cadena)** | Canónica RDKit `[TBD integrar]` preferida para tarea química; `exact_match` cadena reportada como dato (sensible a canonicalización). |
| **Validez / completitud** | `valid_rate`, `incomplete_rate`. Umbrales externos si se definen. |
| **Latencia** | **p50/p95** ms, mismo hardware y execution provider (p. ej. ORT CPU). No solo media. Umbral p95 externo si se define. INT8 dinámico no implica menor latencia. |
| **Tamaño / throughput** | MB en disco (artefactos); throughput (img/s, batch) si aplica. |
| **Clasificador** | Matriz 4×4 + binaria sobre 3 clases mapeadas; `reactions` separada/excluida del binario. |
| **Reproducibilidad** | Fuente/pin/manifiesto/checksums, IDs congelados (seed), hardware/provider/versiones. |"""

PROVENANCE = r"""## 3. Fuentes de datos y procedencia (hechos)

| Suite / uso | Modelo objetivo | Fuente | Identificador / pin | Licencia / procedencia (según manifiesto/fuente) | Artefactos locales |
|---|---|---|---|---|---|
| Golden impreso CLEF | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | dataset commit observado `2fdcdc59`; config `CLEF`, split `test`, 992 | HF agregado **mixed-see-sources**; verificar términos originales CLEF 2012 | `data/benchmark/hf-clef/labels.jsonl`; `data/manifests/benchmark-hf-clef.import.json` |
| Golden impreso JPO | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | config `JPO`, split `test`, 449 | HF agregado **mixed-see-sources**; verificar términos originales JPO/ChemInfty/OSRA | `data/benchmark/hf-jpo/labels.jsonl`; `data/manifests/benchmark-hf-jpo.import.json` |
| Hand-drawn HDM | `ocsr-handdrawn` | Zenodo DECIMER Hand-drawn molecule images | record `7617107`, DOI 10.5281/zenodo.7617107, 5088 | CC-BY-4.0 | `data/benchmark/handdrawn-hdm/labels.jsonl`; `data/manifests/benchmark-handdrawn-hdm.fetch.json` |
| Clasificador (preparado) | `image-classifier` | Zenodo 13378718 | record `13378718`, md5 zip `e7e47f2a16735238104e8dd7ec5e65af` | Registro indica MIT; origen imágenes mixto | `decimer-export import-benchmark --source classifier-13378718` (requiere extracción local) |
| Clasificador 6670746 | `image-classifier` | Zenodo 6670746 | archivo único 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 | No descargado. |

**Mapeo clasificador (13378718, binario):** `one_molecule`→`structure`, `several_molecules`→`structure`, `other`→`no-structure` (incluidas en matriz binaria); `reactions`→excluida del binario, métrica separada `ambiguous_excluded`."""


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
        "Protocolo: variantes comparadas sobre **mismos IDs** cuando aplica (pareado, piloto ⊂ eval ⊂ pool congelado). Latencia **p50/p95** (mismo hardware/provider), no solo media; INT8 dinámico puede no ser más rápido. **Δ vs FP32** = diferencia en pp sobre IDs comunes (misma métrica). `—` = no medido/no disponible.",
        "",
    ]

    def table(title: str, rows: list[str | None]):
        got = [r for r in rows if r]
        if not got:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Variante | n | exact vs label | tf_match (ONNX==TF) | Δtf_match pp vs FP32 | Δexact pp vs FP32 | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.extend(got)
        lines.append("")

    table("5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992)", [
        _row("E1 dynamic INT8 (+TF)", b/"hf-clef/report.e1.tf.json"),
        _row("E3 static-enc + dyn-dec (tf_match por join a TF de E1)", b/"hf-clef/report.e3.tfjoined.json", tf_join_note="[join]"),
    ])
    table("5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449)", [
        _row("E1 dynamic INT8 (+TF)", b/"hf-jpo/report.e1.tf.json"),
        _row("E3 static-enc + dyn-dec (tf_match por join a TF de E1)", b/"hf-jpo/report.e3.tfjoined.json", tf_join_note="[join]"),
    ])
    hdm_fp32 = b/"handdrawn-hdm/report.fp32.tf.limit500.json"
    table("5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra ad hoc `--limit 500`, no decisión formal)", [
        _row("FP32 (+TF)", hdm_fp32, fp32_path=hdm_fp32),
        _row("E1 dynamic INT8 (+TF)", b/"handdrawn-hdm/report.e1.tf.limit500.json", fp32_path=hdm_fp32),
        _row("E3 static-enc + dyn-dec (+TF, calib 32 HDM)", b/"handdrawn-hdm/report.e3.tf.limit500.json", fp32_path=hdm_fp32),
    ])
    # full HDM
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

    lines.append("### 5.5 `image-classifier`")
    lines.append("")
    lines.append("| Suite | Estado | Nota |")
    lines.append("|---|---|---|")
    lines.append("| Zenodo 6670746 | OUT OF SCOPE | No descargar/usar (126.6 GB, archivo único, sin test separado expuesto). |")
    lines.append("| Zenodo 13378718 | Preparado, no ejecutado | Requiere extracción local, export ONNX y wiring; 4×4 + binario (3 clases), `reactions` separada. |")
    lines.append("")
    return "\n".join(lines), pending


LIMITATIONS = r"""## 6. Limitaciones de medición (hechos metodológicos)

- `exact_match` compara cadenas SMILES sin canonicalización RDKit en el harness; diferencias de forma (aromática/Kekulé, orden equivalente) pueden contar como no exactas.
- `tf_match` depende del baseline TF y del pipeline de predicción usado por el harness; reporta fidelidad respecto a esa referencia, no exactitud química canónica.
- `valid` es heurística sin RDKit.
- Latencias son del entorno de ejecución/harness (ver sección de máquina); no son directamente comparables entre máquinas/configuraciones sin controlar hardware, hilos, carga y versiones.
- Subsets HF CLEF/JPO: agregado `mixed-see-sources`; procedencia/licencia original por subset requiere verificación antes de redistribución.
- E3 usa calibración estática; resultados dependen del conjunto/tamaño/distribución de calibración (reportados por experimento)."""


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
        "**Generador:** `decimer_onnx_export.report.generate_report` / `decimer-export report`  ",
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
