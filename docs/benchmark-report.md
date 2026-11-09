# Informe de benchmarks DECIMER → ONNX

> Documento generado automáticamente a partir de reportes JSON/manifiestos en `data/`. No incluye decisiones ni juicios sobre los resultados; reporta mediciones, definiciones y criterios de aceptación definidos para la evaluación.

**Generado (UTC):** `2026-11-09T04:17:18.500594+00:00`  
**Generador:** `decimer_onnx_export.report.generate_report` / `decimer-export report`  
**Rama/commits relevantes:** `main` (workspace fetch/export/runtime/cuantización/benchmark).

---

## 1. Glosario

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
| **McNemar** | Prueba pareada sobre pares discordantes (A acierta/B falla vs inverso) para comparar variantes en el mismo set. |

## 2. Criterios de aceptación (definidos para la evaluación)

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
| **Reproducibilidad** | Fuente/pin/manifiesto/checksums, IDs congelados (seed), hardware/provider/versiones. |

## 3. Fuentes de datos y procedencia (hechos)

| Suite / uso | Modelo objetivo | Fuente | Identificador / pin | Licencia / procedencia (según manifiesto/fuente) | Artefactos locales |
|---|---|---|---|---|---|
| Golden impreso CLEF | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | dataset commit observado `2fdcdc59`; config `CLEF`, split `test`, 992 | HF agregado **mixed-see-sources**; verificar términos originales CLEF 2012 | `data/benchmark/hf-clef/labels.jsonl`; `data/manifests/benchmark-hf-clef.import.json` |
| Golden impreso JPO | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | config `JPO`, split `test`, 449 | HF agregado **mixed-see-sources**; verificar términos originales JPO/ChemInfty/OSRA | `data/benchmark/hf-jpo/labels.jsonl`; `data/manifests/benchmark-hf-jpo.import.json` |
| Hand-drawn HDM | `ocsr-handdrawn` | Zenodo DECIMER Hand-drawn molecule images | record `7617107`, DOI 10.5281/zenodo.7617107, 5088 | CC-BY-4.0 | `data/benchmark/handdrawn-hdm/labels.jsonl`; `data/manifests/benchmark-handdrawn-hdm.fetch.json` |
| Clasificador (preparado) | `image-classifier` | Zenodo 13378718 | record `13378718`, md5 zip `e7e47f2a16735238104e8dd7ec5e65af` | Registro indica MIT; origen imágenes mixto | `decimer-export import-benchmark --source classifier-13378718` (requiere extracción local) |
| Clasificador 6670746 | `image-classifier` | Zenodo 6670746 | archivo único 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 | No descargado. |

**Mapeo clasificador (13378718, binario):** `one_molecule`→`structure`, `several_molecules`→`structure`, `other`→`no-structure` (incluidas en matriz binaria); `reactions`→excluida del binario, métrica separada `ambiguous_excluded`.

## 4. Características de la máquina (recolección automática)

Recolectado automáticamente el `2026-11-09T04:17:18.457373+00:00` mediante `platform`, `lscpu` y `free`.

| Campo | Valor |
|---|---|
| Hostname | `fedora` |
| OS | `Linux 7.2.15-300.fc45.x86_64` |
| Versión kernel/OS | `#1 SMP PREEMPT_DYNAMIC Sun Nov  1 16:11:16 UTC 2026` |
| Arquitectura | `x86_64` |
| Python | `3.11.15` |
| CPU modelo | `Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz` |
| Sockets / núcleos por socket / hilos por núcleo | `1` / `4` / `2` |
| CPUs lógicas | `8` |
| Frecuencia CPU máx/mín (MHz) | `4100.0000` / `400.0000` |
| Cachés L1d/L1i/L2/L3 | `128 KiB (4 instances)` / `128 KiB (4 instances)` / `1 MiB (4 instances)` / `6 MiB (1 instance)` |
| Flags CPU relevantes | `fma sse4_2 avx avx2` |
| RAM total | 15.42 GiB (16555552768 bytes) |
| RAM disponible (al recolectar) | 9.73 GiB |

### Versiones de paquetes (entorno de ejecución)

| Paquete | Versión |
|---|---|
| `Pillow` | `13.0.0` |
| `decimer` | `2.8.1` |
| `numpy` | `1.26.4` |
| `onnx` | `1.16.2` |
| `onnxruntime` | `1.26.0` |
| `sympy` | `1.14.0` |
| `tensorflow` | `2.15.1` |
| `tensorflow-cpu` | `2.15.1` |
| `tf2onnx` | `1.16.1` |
| `typer` | `0.28.0` |

## 5. Resultados

Protocolo: variantes comparadas sobre **mismos IDs** cuando aplica (pareado, piloto ⊂ eval ⊂ pool congelado). Latencia **p50/p95** (mismo hardware/provider), no solo media; INT8 dinámico puede no ser más rápido. **Δ vs FP32** = diferencia en pp sobre IDs comunes (misma métrica). `—` = no medido/no disponible.

### 5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992)

| Variante | n | exact vs label | tf_match (ONNX==TF) | Δtf_match pp vs FP32 | Δexact pp vs FP32 | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 992 | 7 (0.0071) | 838/992 (0.8448) | — | — | 991 (0.9990) | 1 | 5871.20 | 10230.40 | 81.3 | `data/benchmark/hf-clef/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 992 | 6 (0.0060) | 444/992 (0.4476) [join] | — | — | 990 (0.9980) | 2 | 2861.30 | 4919.40 | 84.4 | `data/benchmark/hf-clef/report.e3.tfjoined.json` |

### 5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449)

| Variante | n | exact vs label | tf_match (ONNX==TF) | Δtf_match pp vs FP32 | Δexact pp vs FP32 | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 449 | 9 (0.0200) | 264/449 (0.5880) | — | — | 412 (0.9176) | 37 | 6149.83 | 17156.27 | 81.3 | `data/benchmark/hf-jpo/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 449 | 3 (0.0067) | 123/449 (0.2739) [join] | — | — | 393 (0.8753) | 56 | 2767.13 | 26985.81 | 84.4 | `data/benchmark/hf-jpo/report.e3.tfjoined.json` |

### 5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra ad hoc `--limit 500`, no decisión formal)

| Variante | n | exact vs label | tf_match (ONNX==TF) | Δtf_match pp vs FP32 | Δexact pp vs FP32 | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+TF) | 500 | 198 (0.3960) | 471/500 (0.9420) | (ref) | (ref) | 499 (0.9980) | 1 | 1988.25 | 4504.17 | 301.2 | `data/benchmark/handdrawn-hdm/report.fp32.tf.limit500.json` |
| E1 dynamic INT8 (+TF) | 500 | 195 (0.3900) | 448/500 (0.8960) | -4.60 pp | -0.60 pp | 499 (0.9980) | 1 | 6239.98 | 9709.75 | 81.3 | `data/benchmark/handdrawn-hdm/report.e1.tf.limit500.json` |
| E3 static-enc + dyn-dec (+TF, calib 32 HDM) | 500 | 0 (0.0000) | 0/500 (0.0000) | -94.20 pp | -39.60 pp | 297 (0.5940) | 203 | 25397.42 | 31685.82 | 84.4 | `data/benchmark/handdrawn-hdm/report.e3.tf.limit500.json` |

### 5.4 `ocsr-handdrawn` — HDM completo (5088)

| Variante | n | Estado | Reporte esperado |
|---|---:|---|---|
| E1 dynamic INT8 (+TF) | 5088 | Pendiente / no disponible al generar | `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` |

### 5.5 `image-classifier`

| Suite | Estado | Nota |
|---|---|---|
| Zenodo 6670746 | OUT OF SCOPE | No descargar/usar (126.6 GB, archivo único, sin test separado expuesto). |
| Zenodo 13378718 | Preparado, no ejecutado | Requiere extracción local, export ONNX y wiring; 4×4 + binario (3 clases), `reactions` separada. |

## 7. Archivos de referencia

- Manifiestos fetch/import (trackeados): `data/manifests/benchmark-*.json`.
- Reportes detallados (gitignored, generados localmente): `data/benchmark/**/*.json`.
- Resumen golden printed (trackeado): `data/artifacts/selection/ocsr-printed.hf-golden.json`.
- Selección printed (trackeada, puede contener interpretación): `data/artifacts/selection/ocsr-printed.selection.json`.

## 6. Limitaciones de medición (hechos metodológicos)

- `exact_match` compara cadenas SMILES sin canonicalización RDKit en el harness; diferencias de forma (aromática/Kekulé, orden equivalente) pueden contar como no exactas.
- `tf_match` depende del baseline TF y del pipeline de predicción usado por el harness; reporta fidelidad respecto a esa referencia, no exactitud química canónica.
- `valid` es heurística sin RDKit.
- Latencias son del entorno de ejecución/harness (ver sección de máquina); no son directamente comparables entre máquinas/configuraciones sin controlar hardware, hilos, carga y versiones.
- Subsets HF CLEF/JPO: agregado `mixed-see-sources`; procedencia/licencia original por subset requiere verificación antes de redistribución.
- E3 usa calibración estática; resultados dependen del conjunto/tamaño/distribución de calibración (reportados por experimento).

## 8. Elementos pendientes al generar

- `ocsr-handdrawn` HDM completo n=5088, E1+TF: `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` (no encontrado al generar).
