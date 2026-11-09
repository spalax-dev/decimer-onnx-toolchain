# Informe de benchmarks DECIMER → ONNX

> Documento generado automáticamente a partir de reportes JSON/manifiestos en `data/`. No incluye decisiones ni juicios sobre los resultados; reporta mediciones, definiciones y criterios de aceptación definidos para la evaluación.

**Generado (UTC):** `2026-11-09T02:59:27.992392+00:00`  
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
| **`reactions` (clasificador, métrica separada)** | Categoría tratada como ambigua/excluida del cómputo binario; se reporta aparte (conteo y predicciones), no dentro de TP/TN/FP/FN binarios. |

## 2. Criterios de aceptación (definidos para la evaluación)

Criterios definidos como referencia de medición (sin aplicar veredictos en este documento):

| Criterio | Definición / condición de aceptación |
|---|---|
| **Fidelidad de cuantización/export (estricta)** | `tf_match = 100%` (todas las muestras evaluadas producen salida idéntica al baseline TF bajo el mismo protocolo). |
| **Fidelidad de cuantización/export (tasa)** | `tf_match_rate` medida sobre el conjunto evaluado; se reporta como `tf_match/count`. Cualquier umbral numérico (p. ej., ≥99%, ≥95%) es un parámetro de aceptación externo si se define; este informe solo reporta la tasa observada. |
| **Exactitud vs etiqueta** | `exact_match_rate = exact_match/count` sobre etiquetas del dataset. Aceptación depende de si la comparación usa canonicalización (p. ej., RDKit) y del objetivo (paridad de modelo vs. exactitud de tarea). |
| **Validez / completitud** | Tasas `valid_rate` e `incomplete_rate` reportadas por el harness. Criterios de aceptación (p. ej., incompletos ≤ X%) son parámetros externos si se definen. |
| **Latencia** | `p50/p95` en ms. Criterios de aceptación (p. ej., p95 ≤ X ms en hardware Y) son parámetros externos si se definen. |
| **Clasificador binario** | Matriz de confusión y métricas derivadas calculadas **solo** sobre `one_molecule`, `several_molecules`, `other` según el mapeo explícito; `reactions` se reporta por separado y **excluida** del binario. |
| **Reproducibilidad/procedencia** | Cada conjunto/artefacto reporta fuente, identificador/commit/record, checksums cuando aplican, ruta de manifiesto o labels, y notas de licencia/procedencia. |

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

Recolectado automáticamente el `2026-11-09T02:59:27.906629+00:00` mediante `platform`, `lscpu` y `free`.

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
| RAM disponible (al recolectar) | 6.89 GiB |

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

Columnas: `exact` = exact_match vs etiqueta; `tf_match` = paridad ONNX==TF baseline (ver glosario). `—` = no medido/no disponible en ese reporte.

### 5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992)

| Variante | n | exact vs label | tf_match (ONNX==TF) | valid | incomplete | p50 ms | p95 ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 992 | 7 (0.0071) | 838/992 (0.8448) | 991 (0.9990) | 1 | 5871.20 | 10230.40 | `data/benchmark/hf-clef/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 992 | 6 (0.0060) | 444/992 (0.4476) [join] | 990 (0.9980) | 2 | 2861.30 | 4919.40 | `data/benchmark/hf-clef/report.e3.tfjoined.json` |

### 5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449)

| Variante | n | exact vs label | tf_match (ONNX==TF) | valid | incomplete | p50 ms | p95 ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 449 | 9 (0.0200) | 264/449 (0.5880) | 412 (0.9176) | 37 | 6149.83 | 17156.27 | `data/benchmark/hf-jpo/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 449 | 3 (0.0067) | 123/449 (0.2739) [join] | 393 (0.8753) | 56 | 2767.13 | 26985.81 | `data/benchmark/hf-jpo/report.e3.tfjoined.json` |

### 5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra `--limit 500`)

| Variante | n | exact vs label | tf_match (ONNX==TF) | valid | incomplete | p50 ms | p95 ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+TF) | 500 | 198 (0.3960) | 471/500 (0.9420) | 499 (0.9980) | 1 | 1988.25 | 4504.17 | `data/benchmark/handdrawn-hdm/report.fp32.tf.limit500.json` |
| E1 dynamic INT8 (+TF) | 500 | 195 (0.3900) | 448/500 (0.8960) | 499 (0.9980) | 1 | 6239.98 | 9709.75 | `data/benchmark/handdrawn-hdm/report.e1.tf.limit500.json` |
| E3 static-enc + dyn-dec (+TF, calib 32 HDM) | 500 | 0 (0.0000) | 0/500 (0.0000) | 297 (0.5940) | 203 | 25397.42 | 31685.82 | `data/benchmark/handdrawn-hdm/report.e3.tf.limit500.json` |

### 5.4 `ocsr-handdrawn` — HDM completo (5088)

| Variante | n | Estado | Reporte esperado |
|---|---:|---|---|
| E1 dynamic INT8 (+TF) | 5088 | Pendiente / no disponible al generar | `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` |

### 5.5 `image-classifier`

| Suite | Estado | Nota |
|---|---|---|
| Zenodo 6670746 | No descargado | Archivo único 126.6 GB; sin test separado expuesto por API. |
| Zenodo 13378718 | Preparado, no ejecutado | Requiere extracción local y exportación ONNX del clasificador; binario con `reactions` separada. |

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
