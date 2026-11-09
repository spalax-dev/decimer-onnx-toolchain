# Informe de benchmarks DECIMER → ONNX

> Documento de resultados. No incluye decisiones ni juicios sobre los resultados; reporta mediciones, definiciones y criterios de aceptación definidos para la evaluación.

**Fecha de generación de este informe:** 2026-11-08 (resultados recolectados en esta fecha).  
**Rama/commits relevantes:** `main`, `a979451`–`dc207e3` (workspace, fetch/export/runtime, cuantización, benchmark harness, golden importado).  
**Entorno:** Linux, CPU (TF CPU; ORT CPUExecutionProvider), `uv` workspace (`decimer-onnx-export` incluye TF/DECIMER/ORT; `decimer-onnx` runtime sin TF). Advertencias TF CUDA/TRT/oneDNN y ORT sobre inicializadores no usados se observaron durante ejecuciones; no se interpretan aquí como juicios.

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
| **Latencia p50 / p95 / media (ms)** | Percentiles 50/95 y media del tiempo por muestra medido por el harness (ms), sobre las muestras evaluadas; incluye el costo de inferencia medido en ese bucle (y la ruta de predicción usada). |
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
| **Latencia** | `p50/p95/media` en ms. Criterios de aceptación (p. ej., p95 ≤ X ms en hardware Y) son parámetros externos si se definen. |
| **Clasificador binario** | Matriz de confusión y métricas derivadas calculadas **solo** sobre `one_molecule`, `several_molecules`, `other` según el mapeo explícito; `reactions` se reporta por separado y **excluida** del binario. |
| **Reproducibilidad/procedencia** | Cada conjunto/artefacto reporta fuente, identificador/commit/record, checksums cuando aplican, ruta de manifiesto o labels, y notas de licencia/procedencia. |

## 3. Fuentes de datos y procedencia (hechos)

| Suite / uso | Modelo objetivo | Fuente | Identificador / pin | Licencia / procedencia (según manifiesto/fuente) | Artefactos locales |
|---|---|---|---|---|---|
| DECIMER smoke (4 imágenes) | `ocsr-printed` | `Kohulan/DECIMER-Image_Transformer` | commit `9d0686127d21b572d5e78783874462b542836c6c` | MIT (repo) | `data/benchmark/decimer-smoke/labels.jsonl`; `data/manifests/benchmark-decimer-smoke.fetch.json` (SHA-256 por archivo) |
| Golden impreso CLEF | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | dataset commit observado `2fdcdc59067acb36d84ddc82c1ace24c91d5ad71`; config `CLEF`, split `test`, 992 | Agregado HF **mixed-see-sources**; nota por subset: verificar términos originales CLEF 2012 antes de redistribuir | `data/benchmark/hf-clef/labels.jsonl`; `data/manifests/benchmark-hf-clef.import.json` |
| Golden impreso JPO | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | config `JPO`, split `test`, 449 | Agregado HF **mixed-see-sources**; nota por subset: verificar términos originales JPO/ChemInfty/OSRA antes de redistribuir | `data/benchmark/hf-jpo/labels.jsonl`; `data/manifests/benchmark-hf-jpo.import.json` |
| Hand-drawn HDM | `ocsr-handdrawn` | Zenodo DECIMER Hand-drawn molecule images | record `7617107`, DOI 10.5281/zenodo.7617107, 5088 imágenes + SMILES TSV | CC-BY-4.0 | `data/benchmark/handdrawn-hdm/labels.jsonl`; `data/manifests/benchmark-handdrawn-hdm.fetch.json` (md5 imágenes `7c67bb16bb9819b5bf2699e216ace087`, tsv `3b643bc381be7fe005a7541e9884ca8d`) |
| Clasificador (preparado, no ejecutado aquí) | `image-classifier` | Zenodo 13378718, `dataset_for_image_classifier.zip` | record `13378718`, md5 zip `e7e47f2a16735238104e8dd7ec5e65af` (según API Zenodo) | Registro indica MIT; origen de imágenes mixto (ver registro/fuentes upstream) | Importador: `decimer-export import-benchmark --source classifier-13378718` (requiere zip extraído localmente) |
| Clasificador 6670746 (no descargado) | `image-classifier` | Zenodo 6670746 | record `6670746`, archivo único `DECIMER_Image_classifier_dataset.tar.gz` 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 (registro) | No descargado; API no expone ficheros de test separados. |

**Mapeo clasificador (Zenodo 13378718, explícito para evaluación binaria):**

| Categoría origen | Etiqueta binaria | Incluida en matriz binaria | Reporte |
|---|---|:---:|---|
| `one_molecule` | `structure` | Sí | matriz binaria |
| `several_molecules` | `structure` | Sí | matriz binaria |
| `other` | `no-structure` | Sí | matriz binaria |
| `reactions` | — | No | métrica separada (`ambiguous_excluded`); no entra en TP/TN/FP/FN |

## 4. Artefactos/modelos evaluados (hechos)

| Modelo | Variante | Encoder | Decoder | Vocabulario | Calibración (si aplica) | Baseline TF usado en `tf_match` |
|---|---|---|---|---|---|---|
| `ocsr-printed` | FP32 | `data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx` | `decoder_step.fp32.onnx` | `data/artifacts/tokenizer_SMILES.vocab.json` | n/a | `data/weights/ocsr-printed/DECIMER_model` |
| `ocsr-printed` | E1 dynamic INT8 | `data/artifacts/quant/e1/ocsr-printed/encoder.e1.dynamic.onnx` | `decoder_step.e1.dynamic.onnx` | mismo | n/a | mismo |
| `ocsr-printed` | E3 static-enc + dyn-dec | `data/artifacts/quant/e3/ocsr-printed/encoder.e3.static.onnx` | `decoder_step.e3.dynamic.onnx` | mismo | 2 imágenes fixture (según `ocsr-printed.selection.json` previo / metadata de cuantización) | mismo (paridad E3 calculada por join con resultados TF de E1 en CLEF/JPO) |
| `ocsr-handdrawn` | FP32 | `data/artifacts/fp32/ocsr-handdrawn/encoder.fp32.onnx` | `decoder_step.fp32.onnx` | `data/artifacts/tokenizer_handdrawn.vocab.json` (156 tokens; `token_to_id` idéntico al printed, congelado por separado) | n/a | `data/weights/ocsr-handdrawn/DECIMER_HandDrawn_model` |
| `ocsr-handdrawn` | E1 dynamic INT8 | `data/artifacts/quant/e1/ocsr-handdrawn/encoder.e1.dynamic.onnx` | `decoder_step.e1.dynamic.onnx` | mismo hand-drawn | n/a | mismo |
| `ocsr-handdrawn` | E3 static-enc + dyn-dec | `data/artifacts/quant/e3/ocsr-handdrawn/encoder.e3.static.onnx` | `decoder_step.e3.dynamic.onnx` | mismo hand-drawn | 32 imágenes HDM muestreadas (`data/calibration/ocsr-handdrawn/`, gitignored) | mismo |

Notas de exportación (hechos): encoder `(1,512,512,3) -> (1,256,512)`; decoder step `(tokens[1,T], enc_out[1,256,512]) -> (1,T,156)`, opset 17, dos grafos + bucle greedy Python. Entradas posicionales leak horneadas como inicializadores donde aplica; ORT puede advertir inicializadores no usados en algunos artefactos.

## 5. Resultados

### 5.1 `ocsr-printed` — smoke DECIMER (4 imágenes, MIT, SHA-256 pin)

| Variante | n | exact vs label | tf_match (ONNX==TF) | valid | incomplete | p50 ms | p95 ms | media ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 | 4 | 2/4 (0.500) | 4/4 (1.000) | 4 | 0 | 849.09 | 1464.36 | 962.10 | `data/benchmark/decimer-smoke/report.fp32.json` |
| E3 static-enc + dyn-dec | 4 | 2/4 (0.500) | 4/4 (1.000) | 4 | 0 | 772.99 | 1168.81 | 842.25 | `data/benchmark/decimer-smoke/report.e3.json` |

Detalle por caso (smoke FP32/E3, según reportes): `DECIMER_test_0` y `github_issue_test` presentan `exact_match=false`, `tf_match=true` (diferencia de forma de cadena respecto a etiqueta; salida ONNX igual a TF). `DECIMER_test_1` y `caffeine_upstream` presentan `exact_match=true`, `tf_match=true`.

### 5.2 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, HF import, n=992)

| Variante | n | exact vs label | exact_rate | tf_match (ONNX==TF) | tf_match_rate | valid | valid_rate | incomplete | p50 ms | p95 ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+ TF) | 992 | 7 | 0.0071 | 838 | 0.8448 | 991 | 0.9990 | 1 | 5871.2 | 10230.4 | `data/benchmark/hf-clef/report.e1.tf.json` |
| E3 static-enc + dyn-dec (sin TF en corrida; join a TF de E1) | 992 | 6 | 0.0060 | 444 (join) | 0.4476 | 990 | 0.9980 | 2 | 2861.3 | 4919.4 | `data/benchmark/hf-clef/report.e3.json`, `report.e3.tfjoined.json` |

### 5.3 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, HF import, n=449)

| Variante | n | exact vs label | exact_rate | tf_match (ONNX==TF) | tf_match_rate | valid | valid_rate | incomplete | p50 ms | p95 ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+ TF) | 449 | 9 | 0.0200 | 264 | 0.5880 | 412 | 0.9176 | 37 | 6149.8 | 17156.3 | `data/benchmark/hf-jpo/report.e1.tf.json` |
| E3 static-enc + dyn-dec (sin TF en corrida; join a TF de E1) | 449 | 3 | 0.0067 | 123 (join) | 0.2739 | 393 | 0.8753 | 56 | 2767.1 | 26985.8 | `data/benchmark/hf-jpo/report.e3.json`, `report.e3.tfjoined.json` |

### 5.4 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra `--limit 500`)

| Variante | n | exact vs label | exact_rate | tf_match (ONNX==TF) | tf_match_rate | valid | valid_rate | incomplete | p50 ms | p95 ms | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+ TF) | 500 | 198 | 0.3960 | 471 | 0.9420 | 499 | 0.9980 | 1 | 1988.3 | 4504.2 | `data/benchmark/handdrawn-hdm/report.fp32.tf.limit500.json` |
| E1 dynamic INT8 (+ TF) | 500 | 195 | 0.3900 | 448 | 0.8960 | 499 | 0.9980 | 1 | 6240.0 | 9709.7 | `data/benchmark/handdrawn-hdm/report.e1.tf.limit500.json` |
| E3 static-enc + dyn-dec (+ TF), calib 32 HDM | 500 | 0 | 0.0000 | 0 | 0.0000 | 297 | 0.5940 | 203 | 25397.42 | *(ver reporte JSON)* | `data/benchmark/handdrawn-hdm/report.e3.tf.limit500.json` |

Solapamiento `tf_match` FP32 vs E1 (mismas 500 muestras): ambos 440, solo FP32 31, solo E1 8, ninguno 21.

### 5.5 `ocsr-handdrawn` — HDM completo (5088)

| Variante | n | Estado | Reporte esperado |
|---|---:|---|---|
| E1 dynamic INT8 (+ TF) | 5088 | **Pendiente / en ejecución** al momento de este informe (shell `sh_1bd622b2a001xsXA8005iYPo6b` ejecuta full tras E3-500) | `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` |

### 5.6 `image-classifier`

| Suite | Estado | Nota |
|---|---|---|
| Zenodo 6670746 (126.6 GB, archivo único) | No descargado | Sin ficheros de test expuestos por separado en API; fuera del alcance por tamaño según instrucciones. |
| Zenodo 13378718 (mapeo explícito) | Preparado, no ejecutado | Requiere extracción local de `dataset_for_image_classifier.zip` y exportación ONNX del clasificador; benchmark binario con `reactions` excluida/reportada aparte. |

## 6. Archivos de reporte y manifiestos

- Manifiestos fetch/import (trackeados): `data/manifests/benchmark-*.json`.
- Reportes detallados (gitignored, generados localmente): `data/benchmark/**/*.json`.
- Resumen golden printed (trackeado): `data/artifacts/selection/ocsr-printed.hf-golden.json`.
- Selección printed (trackeada, contiene interpretación de selección): `data/artifacts/selection/ocsr-printed.selection.json`.
- Tokenizadores/vocabs/meta export (trackeados donde aplica): `data/artifacts/tokenizer*.vocab.json`, `data/artifacts/**/export*.meta.json`, `data/artifacts/**/e*.meta.json`.

## 7. Limitaciones de medición (hechos metodológicos)

- `exact_match` compara cadenas SMILES sin canonicalización RDKit en el harness; diferencias de forma (aromática/Kekulé, orden equivalente) pueden contar como no exactas.
- `tf_match` depende del baseline TF y del pipeline de predicción usado por el harness; reporta fidelidad respecto a esa referencia, no exactitud química canónica.
- `valid` es heurística sin RDKit.
- Latencias son de CPU/local del entorno de ejecución y del harness; no son comparables directamente entre máquinas/configuraciones sin controlar hardware, hilos, carga y versiones.
- Subsets HF CLEF/JPO se distribuyen bajo agregado `mixed-see-sources`; la procedencia/licencia original por subset requiere verificación antes de redistribución.
- E3 usa calibración estática; los resultados dependen del conjunto/tamaño de calibración y de la distribución evaluada. Los parámetros de calibración se reportan en cada experimento.
