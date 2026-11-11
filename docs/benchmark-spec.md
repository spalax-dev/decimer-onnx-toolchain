# Especificación de benchmark — DECIMER → ONNX (tres modelos / tres dossiers)

> Documento de diseño/protocolo. Se trabaja con **tres modelos distintos** (distintas fuentes/pesos, tokenizadores, tareas y dominios), cada uno con su **dossier independiente** de evaluación y selección: `ocsr-printed`, `ocsr-handdrawn` e `image-classifier`. **No es un benchmark único** ni hay ranking entre modelos; cada dossier tiene su suite, métricas, criterios de aceptación y decisión de variante ONNX. Lo compartido es solo infraestructura: harness, formato `labels.jsonl`, export/benchmark/reporting.

**Nota de dominio (evidencia observada):** `ocsr-handdrawn` proviene del modelo upstream entrenado para manuscrito (Zenodo 10781330), pero en la cross-evaluación 2×2 (piloto, FP32) rindió competitivo también en impreso (0.712 en impreso vs 0.711 del modelo impreso; 0.719 en manuscrito), mientras `ocsr-printed` degradó en manuscrito (0.677). Por tanto, son modelos distintos con especialización **no exclusiva** (el manuscrito generaliza bien a impreso en esta evaluación).

**Estado:** formalizado (2026-11-08, reframing por modelo/fin).  
**Fuentes operativas:** `docs/benchmark-report.md` (resultados neutrales auto-generados), `data/manifests/*.json`, `data/artifacts/selection/*` (decisiones por dossier/modelo, separadas del informe).

---

## 1. Principio: tres dossiers de calificación

Cada modelo se valida contra **su** suite, con **su** métrica, **su** criterio de aceptación y **su** decisión. Cada fuente de datos pertenece a **exactamente un dossier**.

| Dossier | Modelo | Suite(s) | Pregunta | Decisión que cierra |
|---|---|---|---|---|
| [Impreso](benchmark-dossier-printed.md) | `ocsr-printed` | CLEF + JPO (n=1441) | ¿La variante ONNX reproduce TF y acierta SMILES en impreso? | Variante (E1/E3/FP32) + política de calibración |
| [Manuscrito](benchmark-dossier-handdrawn.md) | `ocsr-handdrawn` | HDM (n=5088) | ¿La variante ONNX reproduce TF y acierta SMILES en manuscrito? | Variante + política de calibración (modelo entrenado para manuscrito; especialización no exclusiva según cross-eval) |
| [Clasificador](benchmark-dossier-classifier.md) | `image-classifier` | Zenodo 13378718 (split definido al importar) | ¿Acierta clasificación structure/no-structure según mapeo definido? | Export ONNX + wiring + variante |

**No aplica:** matriz 2×2 cross-domain como requisito (no hay router que envíe la misma imagen a ambos modelos OCSR). Las celdas cross-domain quedan, si acaso, como **chequeo opcional de robustez**, no como parte del dossier. **No** hay tabla comparativa única ni pooled entre dossiers/dominios.

## 2. Fuentes (cada una en un solo dossier)

| ID fuente | Dossier | n / alcance | Licencia / procedencia | Estado | Manifiesto / labels |
|---|---|---:|---|---|---|
| `hf-clef` | Impreso | 992 | Keylab/OCSR-Benchmarks, commit obs. `2fdcdc59`; HF **mixed-see-sources**; verificar términos originales CLEF 2012 | **IN** | `data/manifests/benchmark-hf-clef.import.json`, `data/benchmark/hf-clef/labels.jsonl` |
| `hf-jpo` | Impreso | 449 | idem agregado; verificar términos originales JPO/ChemInfty/OSRA | **IN** | `data/manifests/benchmark-hf-jpo.import.json`, `data/benchmark/hf-jpo/labels.jsonl` |
| `handdrawn-hdm` | Manuscrito | 5088 | Zenodo 7617107, CC-BY-4.0, md5 verificados | **IN** | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |
| `classifier-13378718` | Clasificador | por split extraído | Zenodo 13378718, md5 zip `e7e47f2aa4af4b79874a19382cec06da` (Zenodo API); registro MIT, origen imágenes mixto (verificar) | **IN** (requiere extracción local + export ONNX) | importador `--source classifier-13378718` |
| Zenodo **6670746** | — | 126.6 GB, archivo único | CC-BY-4.0; sin test separado expuesto | **OUT OF SCOPE (formalizado)** — no se descarga ni usa | — |
| `decimer-smoke` | Impreso (sanity opcional) | 4 | DECIMER-Image_Transformer `@9d06861`, MIT, SHA-256 | **IN opcional**, fuera del informe principal | `data/manifests/benchmark-decimer-smoke.fetch.json` |

## 3. Regla de `n` (estandarización)

- **`n` se estandariza por dossier y por conjunto de IDs congelado.** Nunca “todo a 1441” entre dossiers.
- **Dentro de un dossier:** todas las variantes comparadas ven los **mismos IDs** (comparación pareada). Congelar subconjunto de iteración con **seed + IDs en manifiesto**; el conjunto completo del dossier es el resultado final.
  - **Impreso:** pool/frozen n=1400 (CLEF+JPO, seed `20260308`) → `eval=1000`, `pilot=300`, `remainder=400` (`data/manifests/frozen/printed.frozen.json`).
  - **Manuscrito:** frozen n=1400 (HDM, seed `20260308`) → `eval=1000`, `pilot=300`, `remainder=400` (`data/manifests/frozen/handdrawn.frozen.json`).
  - **Clasificador:** frozen n=4000 estratificado (seed `20260308`) → `eval=1000`, `pilot=300` (`data/manifests/frozen/classifier.frozen.json`). `reactions` excluida del binario y reportada aparte.

## 4. Contratos compartidos (infraestructura)

### 4.1 Protocolo de medición pareado (por variante)
Cada variante se mide sobre el **mismo set congelado / mismos IDs** (piloto ⊂ eval ⊂ pool; nunca mezclar sets entre variantes). Por cada corrida registrar:
- **Precisión/tarea (primaria):** métrica definida por dossier. **OCSR:** exactitud SMILES **canónica RDKit** (`canonical_match`, **primaria**, integrada) + `exact_match` cadena (secundaria); fidelidad `tf_match`/`tf_canonical_match` ONNX↔TF. **Clasificador:** binary accuracy (primaria). **Piso absoluto: ≥70%** en métrica primaria (eval congelado). Reportar **Δ absoluto en pp contra FP32** (misma métrica, mismos IDs).
- **Latencia:** **p50 y p95** en el **mismo hardware** y **mismo execution provider** (p. ej. ORT `CPUExecutionProvider`), registrados. No reportar solo media; la cola (p95) es relevante.
- **Tamaño en disco:** MB de los artefactos del modelo (encoder+decoder u equivalente), por variante.
- **Throughput:** si se procesan lotes, reportar lotes/s o imágenes/s con tamaño de lote `[si aplica]`.
- **Comparación de variantes (pareada):** **McNemar** sobre pares discordantes + Δ pp con **IC bootstrap**; reportar `n` + IC.
- **Caché:** predicciones por `(model_key, variant_key, image_id)` para no recalcular (`data/cache/predictions/`, gitignored).

**Advertencia de medición:** la cuantización dinámica (INT8) **no siempre es más rápida** (el overhead de cuantizar activaciones en runtime puede anular/peorar la ganancia). La latencia se mide, no se asume.

### 4.2 Etiquetas / métricas por bloque
- **OCSR (impreso/manuscrito):** **`canonical_match` RDKit (primaria, calidad química)** sobre pares comparables + `exact_match` cadena (secundaria), `tf_match`/`tf_canonical_match` (fidelidad ONNX↔TF), `valid`/`incomplete`, latencia p50/p95, tamaño, throughput si aplica.
- **Clasificador:** **accuracy binaria (primaria)** sobre 3 clases mapeadas (`one/several_molecule→structure`, `other/rest→no-structure`), **matriz** `[[TN,FP],[FN,TP]]`, con **`reactions` excluida del binario** y reportada aparte (`ambiguous_excluded`). Fidelidad ONNX↔baseline/paridad TF.
- **Piso absoluto (obligatorio, todos los modelos/variantes aceptables):** **accuracy ≥ 70%** en la métrica primaria del dossier, sobre el **split `eval` congelado** (mismos IDs). OCSR = `canonical_match_rate` (RDKit, comparables). Clasificador = binary accuracy (no-reacción). Variantes < 70% **no se aceptan**.
- **Criterios adicionales (resueltos por evidencia):** para variantes comprimidas se exige cumplir el piso ≥70% de forma **robusta** (no solo borde en `eval`; validado contra `remainder`/held-out cuando aplica). `Δ vs FP32` y p95 se registran; la selección prioriza cumplir el piso y fidelidad/calidad (y compresión cuando no degrada el piso). McNemar/IC bootstrap pareado quedan como mejora de tooling futura (el reporte registra Δpp y `n` pareado).

### 4.2 Preprocesado (registrado por modelo)
Registrar entrada, resize/normalización/canales, ruta TF baseline vs runtime ONNX usada en `tf_match`, versión. Control **FP32+TF** antes de concluir sobre cuantización (separa export/preprocess vs cuantización).

## 5. Reporte

- Reporte neutral auto-generado por `dtx report` (`docs/benchmark-report.md`) desde `data/benchmark/**/*.json` y manifiestos: incluye **todos los resultados** por dossier/split (eval/pilot/remainder), variantes no seleccionadas incluidas; Imago excluido (experimento aparte en `docs/experiments/`).
- Comparación pareada (mismos IDs) dentro de dossier, Δpp vs FP32, métricas primarias/secundarias, p50/p95, tamaños. Secciones por dossier/modelo y fuentes históricas cuando aplican.

## 6. Decisiones (consolidadas)

| Dossier | Variante seleccionada | Evidencia (métrica primaria, eval congelado) | Nota |
|---|---|---|---|
| Impreso `ocsr-printed` | **FP32** | 0.7041 canónico (614/872, n=1000) ≥70% | Ninguna variante comprimida/mixta probada (E1/E3, mixes enc/dec, e1s, particiones por bloques) fue robustamente ≥70% en held-out. |
| Manuscrito `ocsr-handdrawn` | **FP32** | 0.7300 canónico (576/789, n=1000) ≥70% | Cumple piso con mejor calidad/fidelidad; E1 cumplió accuracy en eval pero con tradeoff de latencia/fidelidad. |
| Clasificador `image-classifier` | **E4 FP16** | 0.7679 acc binaria (n=1000, no-reaction) ≥70% | Paridad TF 0.998, 7.78 MB (-49%). INT8 Conv (E1-all/E2) <70%; E1s sin compresión relevante. |

**Matriz de calificación ponderada (1–100):** ver [`docs/selection-scoring.md`](selection-scoring.md) para criterios, pesos por dossier, puntajes por variante y argumento de selección.

Artefactos seleccionados consolidados en `chemistry-draw-ocsr/` (ignorado en este repo) para publicación en HF/`chemistry_draw`.

## 7. Estado actual y conformidad con el protocolo (realidad)

### Ejecutado conforme al protocolo (núcleo)
- **Conjuntos congelados y pareo:** manifiestos frozen (seed `20260308`) y splits `eval`/`pilot`/`remainder` por dossier (`data/manifests/frozen/*.frozen.json`); comparaciones sobre **mismos IDs** dentro del mismo split (`--frozen-manifest ... --split ...`).
- **Métricas:** OCSR con **`canonical_match` RDKit** (primaria, comparables), `exact_match` (secundaria) y fidelidad `tf_match`/`tf_canonical_match` vs TF; clasificador con **accuracy binaria** (no-reacción), matriz, paridad TF y `reactions` separada.
- **Selección y registro:** piso **≥70%** (métrica primaria, eval congelado) con criterio de robustez (validación en `remainder` cuando aplica). **Todos** los resultados (incluidos no seleccionados) en `data/benchmark/**/*.json` y en `docs/benchmark-report.md` auto-generado (`dtx report`); Imago excluido del reporte y documentado en `docs/experiments/`.
- **Latencia/caché:** `p50/p95`, `n_measured` y `cache_hits` reportados; caché por `(model_key, variant_key, image_id)`. Referencias FP32+TF medidas y Δpp vs FP32 registrados.

### Desviaciones / matices respecto al ideal
- **Latencia y contención de CPU:** algunas corridas (pilotos/remainders) se ejecutaron con otros jobs en paralelo, lo que elevó `p50/p95` y las hace **no comparables** como latencia; la **accuracy no se ve afectada**. Las mediciones en solitario (p. ej. evals/remainders sin contención) son las representativas para comparar latencia.
- **Estadística pareada formal no automatizada:** no se calculan automáticamente **McNemar ni IC bootstrap** sobre discordancias; se reportan tasas, Δpp y `n` pareado. (Mejora de tooling pendiente.)
- **Evidencia histórica:** coexisten corridas históricas previas al protocolo frozen (p. ej. CLEF/JPO 1441, muestras HDM ad hoc) etiquetadas como tal; las decisiones consolidadas se basan en splits frozen.
- **Criterios finos:** umbrales adicionales (Δpp/p95 como gate numérico duro único) se resolvieron por evidencia/robustez y registro de métricas, no como un único umbral pre-fijado global más allá del piso ≥70%.

### Incidentes de ejecución (mitigados)
- **Reintentos por carga ONNX transitoria:** fallos `NoSuchFile` al iniciar algunas corridas (p. ej. variantes `sel*`) pese a existir los artefactos; se reintentó y las ejecuciones completadas quedaron registradas.
- **Viabilidad/runtime:** E4 FP16 OCSR no viable en ORT CPU (error de broadcast en atención del decoder) → descartada. E3 (QDQ estático enc.) produjo degradación severa en OCSR → rechazada.
- **Imago:** crashes nativos intermitentes mitigados con runner por subproceso por imagen (`scripts/benchmark_imago_resilient.py`); resultados solo en `docs/experiments/` (fuera del reporte).
- **Avisos de cuantización:** warnings de pre-procesamiento de ONNX Runtime durante cuantización dinámica; no impidieron generar/evaluar artefactos.

### Pendiente (mejoras de tooling/ingeniería)
- McNemar/IC bootstrap automatizados; gates finos de p95 si se fijan; empaquetado canónico del toolchain (`selection.json` unificado/`default`, NOTICE/manifest de bundle); tareas M1/M6 (check/CI, probe, hardening).
