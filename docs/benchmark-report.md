# Informe de benchmarks DECIMER → ONNX

> Documento generado automáticamente a partir de reportes JSON/manifiestos en `data/`. No incluye decisiones ni juicios sobre los resultados; reporta mediciones, definiciones y criterios de aceptación definidos para la evaluación.

**Generado (UTC):** `2026-11-09T22:21:34.141214+00:00`  
**Generador:** `toolchain.report.generate_report` / `dtx report` (alias: `decimer-export report`)  
**Rama/commits relevantes:** `main` (workspace fetch/export/runtime/cuantización/benchmark).

---

## 1. Glosario

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
| **McNemar** | Prueba pareada sobre discordancias (A acierta/B falla vs inverso) en el mismo set. | — | `[pendiente implementar en reporte/harness]`. |

## 2. Criterios de aceptación (definidos para la evaluación)

Criterios definidos como referencia de medición (sin aplicar veredictos en este documento):

| Criterio | Definición / condición de aceptación |
|---|---|
| **Comparación pareada** | Todas las variantes se miden sobre el **mismo set/IDs congelados** (piloto ⊂ eval ⊂ pool). Δ vs FP32 en pp sobre mismos IDs. |
| **Fidelidad (tasa / estricta)** | `tf_match_rate = tf_match/count` (ONNX==TF). Estricto: 100%. Umbral numérico (`≥ [TBD]%`) es parámetro externo si se define. |
| **Δ vs FP32 (precisión/fidelidad)** | Diferencia absoluta en pp vs FP32, misma métrica e IDs. Umbral `≤ [TBD] pp` es parámetro externo si se define. |
| **Exactitud (canónica / cadena)** | Canónica RDKit (`canonical_match`) integrada en benchmark OCSR (cuando ambos SMILES parsean). Preferida para tarea química. `exact_match` string (`predicho == esperado`, sin canonicalizar) se reporta como dato. |
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
| Clasificador (preparado) | `image-classifier` | Zenodo 13378718 | record `13378718`, md5 zip `e7e47f2aa4af4b79874a19382cec06da` (Zenodo API) | Registro indica MIT; origen imágenes mixto | `decimer-export import-benchmark --source classifier-13378718` (requiere extracción local) |
| Clasificador 6670746 | `image-classifier` | Zenodo 6670746 | archivo único 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 | No descargado. |

**Mapeo clasificador (13378718, binario):** `one_molecule`→`structure`, `several_molecules`→`structure`, `other`→`no-structure` (incluidas en matriz binaria); `reactions`→excluida del binario, métrica separada `ambiguous_excluded`.

## 4. Características de la máquina (recolección automática)

Recolectado automáticamente el `2026-11-09T22:21:34.079774+00:00` mediante `platform`, `lscpu` y `free`.

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
| RAM disponible (al recolectar) | 9.31 GiB |

### Versiones de paquetes (entorno de ejecución)

| Paquete | Versión |
|---|---|
| `Pillow` | `13.0.0` |
| `decimer` | `2.8.1` |
| `numpy` | `1.26.4` |
| `onnx` | `1.16.2` |
| `onnxruntime` | `1.26.0` |
| `rdkit` | `2026.9.1` |
| `sympy` | `1.14.0` |
| `tensorflow` | `2.15.1` |
| `tensorflow-cpu` | `2.15.1` |
| `tf2onnx` | `1.16.1` |
| `typer` | `0.28.0` |

## 5. Resultados

Protocolo: variantes comparadas sobre **mismos IDs** cuando aplica (pareado, piloto ⊂ eval ⊂ pool congelado). Latencia **p50/p95** (mismo hardware/provider), no solo media; INT8 dinámico puede no ser más rápido. **Δ vs FP32** = diferencia en pp sobre IDs comunes (misma métrica). `—` = no medido/no disponible. Ver glosario (§1) para significado de columnas.

### 5.P1 `ocsr-printed` — piloto congelado (pilot, TF+cache) (frozen n=1400, eval=1000, pilot=300, seed=20260308, source=`data/manifests/frozen/printed.frozen.json`)

| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+TF) | 300 | **187/263 (0.7110)** | 1 (0.0033) | 248/300 (0.8267) / can 230/263 (0.875) | (ref) | (ref) | (ref) | 290 (0.9667) | 10 | — (cache, sin latencia fresca) | — | 301.2 | `data/benchmark/pilot/printed.fp32.pilot300.json` |
| E1 dynamic INT8 all (+TF) | 300 | **182/266 (0.6842)** | 1 (0.0033) | 230/300 (0.7667) / can 219/262 (0.836) | -2.68 pp | -6.00 pp | +0.00 pp | 292 (0.9733) | 8 | — (cache, sin latencia fresca) | — | 81.3 | `data/benchmark/pilot/printed.int8-dyn-all.pilot300.json` |

### 5.P2 `ocsr-handdrawn` — piloto congelado (pilot, TF+cache) (frozen n=1400, eval=1000, pilot=300, seed=20260308, source=`data/manifests/frozen/handdrawn.frozen.json`)

| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+TF) | 300 | **210/292 (0.7192)** | 121 (0.4033) | 279/300 (0.9300) / can 273/291 (0.938) | (ref) | (ref) | (ref) | 299 (0.9967) | 1 | — (cache, sin latencia fresca) | — | 301.2 | `data/benchmark/pilot/handdrawn.fp32.pilot300.json` |
| E1 dynamic INT8 all (+TF) | 300 | **204/293 (0.6962)** | 116 (0.3867) | 266/300 (0.8867) / can 263/290 (0.907) | -2.29 pp | -4.33 pp | -1.67 pp | 299 (0.9967) | 1 | — (cache, sin latencia fresca) | — | 81.3 | `data/benchmark/pilot/handdrawn.int8-dyn-all.pilot300.json` |

### 5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992, evidencia histórica)

| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 992 | — | 7 (0.0071) | 838/992 (0.8448) | — | — | — | 991 (0.9990) | 1 | 5871.20 | 10230.40 | 81.3 | `data/benchmark/hf-clef/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 992 | — | 6 (0.0060) | 444/992 (0.4476) [join] | — | — | — | 990 (0.9980) | 2 | 2861.30 | 4919.40 | 84.4 | `data/benchmark/hf-clef/report.e3.tfjoined.json` |

### 5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449, evidencia histórica)

| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 449 | — | 9 (0.0200) | 264/449 (0.5880) | — | — | — | 412 (0.9176) | 37 | 6149.83 | 17156.27 | 81.3 | `data/benchmark/hf-jpo/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 449 | — | 3 (0.0067) | 123/449 (0.2739) [join] | — | — | — | 393 (0.8753) | 56 | 2767.13 | 26985.81 | 84.4 | `data/benchmark/hf-jpo/report.e3.tfjoined.json` |

### 5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (muestra ad hoc `--limit 500`, no decisión formal)

| Variante | n | **canonical RDKit (primaria, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | tamaño MB (enc+dec) | Reporte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+TF) | 500 | — | 198 (0.3960) | 471/500 (0.9420) | (ref) | (ref) | (ref) | 499 (0.9980) | 1 | 1988.25 | 4504.17 | 301.2 | `data/benchmark/handdrawn-hdm/report.fp32.tf.limit500.json` |
| E1 dynamic INT8 (+TF) | 500 | — | 195 (0.3900) | 448/500 (0.8960) | — | -4.60 pp | -0.60 pp | 499 (0.9980) | 1 | 6239.98 | 9709.75 | 81.3 | `data/benchmark/handdrawn-hdm/report.e1.tf.limit500.json` |
| E3 static-enc + dyn-dec (+TF, calib 32 HDM) | 500 | — | 0 (0.0000) | 0/500 (0.0000) | — | -94.20 pp | -39.60 pp | 297 (0.5940) | 203 | 25397.42 | 31685.82 | 84.4 | `data/benchmark/handdrawn-hdm/report.e3.tf.limit500.json` |

### 5.4 `ocsr-handdrawn` — HDM completo (5088)

| Variante | n | Estado | Reporte esperado |
|---|---:|---|---|
| E1 dynamic INT8 (+TF) | 5088 | Pendiente / no disponible al generar | `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` |

### 5.5 `image-classifier` (Zenodo 13378718, congelado)

Frozen: n=4000, eval=1000, pilot=300, seed=20260308, clases 10k={'one_molecule': 1000, 'other': 1000, 'reactions': 1000, 'several_molecules': 1000}. Binaria sobre `one/several_molecule→structure`, `other→no-structure`; `reactions` excluida del binario y reportada aparte. Tamaño: FP32 16M, E1 int8-dyn-all 4.4M (`data/artifacts/*/image-classifier/*.onnx`).

| Variante (split) | total filas | Binaria acc | Matriz `[[TP,FN],[FP,TN]]` | Δacc vs FP32 | Paridad TF (incl. reactions) | reactions (separada) | p50 ms | p95 ms | Reporte |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| `fp32` (`pilot`) | 300 | 0.7361 (n=216) | `[[90, 56], [1, 69]]` | (ref) | 300/300 (1.0000) | 84 → struct 55 / no-struct 29 | — (cache, sin latencia fresca) | — | `data/benchmark/pilot/classifier.fp32.pilot300.json` |
| `int8-dyn-all` (`pilot`) | 300 | 0.3796 (n=216) | `[[12, 134], [0, 70]]` | -35.65 pp | 166/300 (0.5533) | 84 → struct 0 / no-struct 84 | 321.64 | 572.67 | `data/benchmark/pilot/classifier.int8-dyn-all.pilot300.json` |

| Suite | Estado | Nota |
|---|---|---|
| Zenodo 6670746 | OUT OF SCOPE | No descargar/usar (126.6 GB, archivo único, sin test separado expuesto). |
| Zenodo 13378718 | Piloto ejecutado (FP32/E1); eval-1000 pendiente | 4→binario mapeado explícitamente; `reactions` separada. |

## 7. Archivos de referencia

- Manifiestos fetch/import (trackeados): `data/manifests/benchmark-*.json`.
- Reportes detallados (gitignored, generados localmente): `data/benchmark/**/*.json`.
- Resumen golden printed (trackeado): `data/artifacts/selection/ocsr-printed.hf-golden.json`.
- Selección printed (trackeada, puede contener interpretación): `data/artifacts/selection/ocsr-printed.selection.json`.

## 6. Limitaciones de medición (hechos metodológicos)

- `exact_match` (string) compara `predicho == esperado` **sin canonicalización RDKit**. `canonical_match` (RDKit) compara SMILES canónicos cuando ambos parsean (integrado en benchmark OCSR; puede no estar en reportes históricos). Diferencias equivalentes de representación pueden contar como no exactas en `exact_match`.
- `tf_match` mide fidelidad **ONNX↔TF** según el baseline/pipeline del harness; no es exactitud química canónica frente a la etiqueta.
- `valid`/`incomplete` son heurísticas sin RDKit (aproximación).
- Latencias son del entorno de ejecución/harness (máquina/provider/carga/hilos/versiones). Reportar **p50/p95**; no usar solo media. `cache_hits` invalida latencia fresca. INT8 dinámico puede ser más lento que FP32 en CPU.
- Δ en pp calculado sobre la misma métrica e IDs cuando hay datos pareados; si falta pareo, puede indicarse agregado.
- Subsets HF CLEF/JPO: agregado `mixed-see-sources`; verificar términos originales antes de redistribución.
- Cuantización estática (E3/QDQ) depende fuertemente del set de calibración (tamaño/distribución/holdout); resultados no generalizan sin protocolo controlado.
- Clasificador: binario solo sobre 3 clases mapeadas; `reactions` excluida y reportada aparte (`ambiguous_excluded`).

## 8. Elementos pendientes al generar

- `ocsr-handdrawn` HDM completo n=5088, E1+TF: `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` (no encontrado al generar).
