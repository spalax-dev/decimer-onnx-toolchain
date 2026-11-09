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


def _row(label: str, report_path: Path, *, tf_join_note: str = "") -> str | None:
    r = _load_json(report_path)
    if not r:
        return None
    n = r.get("count")
    ex = r.get("exact_match")
    exr = r.get("exact_match_rate")
    tfm = r.get("tf_match")
    tfmr = r.get("tf_match_rate")
    tf_cell = _fmt_rate(tfm, n)
    if tfm is None:
        tf_cell = "—"
    elif tf_join_note:
        tf_cell = f"{tfm}/{n} ({tfmr:.4f}) {tf_join_note}" if tfmr is not None else f"{tfm}/{n} {tf_join_note}"
    valid = r.get("valid")
    vr = r.get("valid_rate")
    inc = r.get("incomplete")
    lat = r.get("latency_ms") or {}
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    return (
        f"| {label} | {n} | {ex} ({exr:.4f}) | {tf_cell} | {valid} ({vr:.4f}) | {inc} | "
        f"{_fmt_ms(lat.get('p50'))} | {_fmt_ms(lat.get('p95'))} | `{rep}` |"
    )


GLOSSARY = r"""## 1. Glosario

| Término | Definición |
|---|---|
| **OCSR** | Optical Chemical Structure Recognition: reconocimiento de estructuras químicas desde imágenes para producir una representación (aquí, SMILES). |
| **SMILES** | Simplified Molecular Input Line Entry System: cadena textual que representa una estructura molecular. |
| **Modelo TF (baseline)** | Modelo DECIMER original ejecutado en TensorFlow (SavedModel), usado como referencia de comportamiento. |
| **Modelo ONNX** | Modelo convertido a ONNX (encoder + `decoder_step`) ejecutado con ONNX Runtime (ORT). |
| **FP32** | Exportación ONNX en precisión flotante de 32 bits. |
| **E1 (dynamic INT8)** | Cuantización dinámica INT8 (pesos/ops cuantizados según configuración de ORT; sin calibración). |
| **E3 (hybrid static-enc + dynamic-dec)** | Encoder cuantizado estático INT8/QDQ con calibración + decoder cuantizado dinámico INT8. |
| **Calibración** | Proceso de alimentación de muestras representativas para cuantización estática (rango/escala). El conjunto y tamaño de calibración se reportan por experimento. |
| **Paridad TF↔ONNX (export)** | Equivalencia de comportamiento entre el modelo TF baseline y el modelo ONNX bajo el mismo protocolo de prueba. |
| **`exact_match` (vs etiqueta)** | Proporción de casos donde el SMILES predicho es **idéntico como cadena** al SMILES etiqueta del dataset (`got == expected`). |
| **`tf_match`** | Proporción de casos donde el SMILES (o salida comparable) del pipeline ONNX es **idéntico** al del pipeline TF baseline para la misma entrada, bajo el harness de benchmark. En los experimentos OCSR con `--tf-model-dir`, el harness compara la predicción ONNX (preprocesado/runtime del harness ONNX) con la predicción TF de referencia; reporta `tf_smiles` y `tf_match` por caso. |
| **`valid`** | Conteo/tasa según la heurística del harness: predicción no vacía y no marcada como incompleta (heurística sin RDKit). |
| **`incomplete`** | Predicción marcada como incompleta por el decodificador/harness (p. ej., secuencia no finalizada según el criterio del tokenizador/decodificación). |
| **Latencia p50 / p95 (ms)** | Percentiles 50/95 del tiempo por muestra medido por el harness (ms), sobre las muestras evaluadas; incluye el costo de inferencia medido en ese bucle. |
| **Matriz de confusión binaria (clasificador)** | Conteo TP/TN/FP/FN sobre las clases binarias `structure` / `no-structure`, calculada solo sobre categorías mapeadas como incluidas en binario. |
| **`reactions` (clasificador, métrica separada)** | Categoría tratada como ambigua/excluida del cómputo binario; se reporta aparte (conteo y predicciones), no dentro de TP/TN/FP/FN binarios. |"""

CRITERIA = r"""## 2. Criterios de aceptación (definidos para la evaluación)

Criterios definidos como referencia de medición (sin aplicar veredictos en este documento):

| Criterio | Definición / condición de aceptación |
|---|---|
| **Fidelidad de cuantización/export (estricta)** | `tf_match = 100%` (todas las muestras evaluadas producen salida idéntica al baseline TF bajo el mismo protocolo). |
| **Fidelidad de cuantización/export (tasa)** | `tf_match_rate` medida sobre el conjunto evaluado; se reporta como `tf_match/count`. Cualquier umbral numérico (p. ej., ≥99%, ≥95%) es un parámetro de aceptación externo si se define; este informe solo reporta la tasa observada. |
| **Exactitud vs etiqueta** | `exact_match_rate = exact_match/count` sobre etiquetas del dataset. Aceptación depende de si la comparación usa canonicalización (p. ej., RDKit) y del objetivo (paridad de modelo vs. exactitud de tarea). |
| **Validez / completitud** | Tasas `valid_rate` e `incomplete_rate` reportadas por el harness. Criterios de aceptación (p. ej., incompletos ≤ X%) son parámetros externos si se definen. |
| **Latencia** | `p50/p95` en ms. Criterios de aceptación (p. ej., p95 ≤ X ms en hardware Y) son parámetros externos si se definen. |
| **Clasificador binario** | Matriz de confusión y métricas derivadas calculadas **solo** sobre `one_molecule`, `several_molecules`, `other` según el mapeo explícito; `reactions` se reporta por separado y **excluida** del binario. |
| **Reproducibilidad/procedencia** | Cada conjunto/artefacto reporta fuente, identificador/commit/record, checksums cuando aplican, ruta de manifiesto o labels, y notas de licencia/procedencia. |"""

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
    lines = ["## 5. Resultados", "", "Columnas: `exact` = exact_match vs etiqueta; `tf_match` = paridad ONNX==TF baseline (ver glosario). `—` = no medido/no disponible en ese reporte.", ""]

    def table(title: str, rows: list[str | None]):
        got = [r for r in rows if r]
        if not got:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Variante | n | exact vs label | tf_match (ONNX==TF) | valid | incomplete | p50 ms | p95 ms | Reporte |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
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
    table("5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra `--limit 500`)", [
        _row("FP32 (+TF)", b/"handdrawn-hdm/report.fp32.tf.limit500.json"),
        _row("E1 dynamic INT8 (+TF)", b/"handdrawn-hdm/report.e1.tf.limit500.json"),
        _row("E3 static-enc + dyn-dec (+TF, calib 32 HDM)", b/"handdrawn-hdm/report.e3.tf.limit500.json"),
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
    lines.append("| Zenodo 6670746 | No descargado | Archivo único 126.6 GB; sin test separado expuesto por API. |")
    lines.append("| Zenodo 13378718 | Preparado, no ejecutado | Requiere extracción local y exportación ONNX del clasificador; binario con `reactions` separada. |")
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
