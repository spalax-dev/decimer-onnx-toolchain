# Especificación de benchmark — DECIMER → ONNX (tres dossiers)

> Documento de diseño/protocolo. **No es un benchmark único: son tres dossiers independientes**, uno por modelo, cada uno con su suite, métricas, criterios de aceptación y decisión de variante ONNX. No hay ranking entre modelos. Lo compartido es solo infraestructura: harness, formato `labels.jsonl`, export ONNX, plantilla de criterios, generador de reportes.

**Estado:** formalizado (2026-11-08, reframing por modelo/fin).  
**Fuentes operativas:** `docs/benchmark-report.md` (resultados neutrales auto-generados), `data/manifests/*.json`, `data/artifacts/selection/*` (decisiones por dossier/modelo, separadas del informe).

---

## 1. Principio: tres dossiers de calificación

Cada modelo se valida contra **su** suite, con **su** métrica, **su** criterio de aceptación y **su** decisión. Cada fuente de datos pertenece a **exactamente un dossier**.

| Dossier | Modelo | Suite(s) | Pregunta | Decisión que cierra |
|---|---|---|---|---|
| [Impreso](benchmark-dossier-printed.md) | `ocsr-printed` | CLEF + JPO (n=1441) | ¿La variante ONNX reproduce TF y acierta SMILES en impreso? | Variante (E1/E3/FP32) + política de calibración |
| [Manuscrito](benchmark-dossier-handdrawn.md) | `ocsr-handdrawn` | HDM (n=5088) | ¿La variante ONNX reproduce TF y acierta SMILES en manuscrito? | Variante + política de calibración |
| [Clasificador](benchmark-dossier-classifier.md) | `image-classifier` | Zenodo 13378718 (split definido al importar) | ¿Acierta clasificación structure/no-structure según mapeo definido? | Export ONNX + wiring + variante |

**No aplica:** matriz 2×2 cross-domain como requisito (no hay router que envíe la misma imagen a ambos modelos OCSR). Las celdas cross-domain quedan, si acaso, como **chequeo opcional de robustez**, no como parte del dossier. **No** hay tabla comparativa única ni pooled entre dossiers/dominios.

## 2. Fuentes (cada una en un solo dossier)

| ID fuente | Dossier | n / alcance | Licencia / procedencia | Estado | Manifiesto / labels |
|---|---|---:|---|---|---|
| `hf-clef` | Impreso | 992 | Keylab/OCSR-Benchmarks, commit obs. `2fdcdc59`; HF **mixed-see-sources**; verificar términos originales CLEF 2012 | **IN** | `data/manifests/benchmark-hf-clef.import.json`, `data/benchmark/hf-clef/labels.jsonl` |
| `hf-jpo` | Impreso | 449 | idem agregado; verificar términos originales JPO/ChemInfty/OSRA | **IN** | `data/manifests/benchmark-hf-jpo.import.json`, `data/benchmark/hf-jpo/labels.jsonl` |
| `handdrawn-hdm` | Manuscrito | 5088 | Zenodo 7617107, CC-BY-4.0, md5 verificados | **IN** | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |
| `classifier-13378718` | Clasificador | por split extraído | Zenodo 13378718, md5 zip `e7e47f2a16735238104e8dd7ec5e65af`; registro MIT, origen imágenes mixto (verificar) | **IN** (requiere extracción local + export ONNX) | importador `--source classifier-13378718` |
| Zenodo **6670746** | — | 126.6 GB, archivo único | CC-BY-4.0; sin test separado expuesto | **OUT OF SCOPE (formalizado)** — no se descarga ni usa | — |
| `decimer-smoke` | Impreso (sanity opcional) | 4 | DECIMER-Image_Transformer `@9d06861`, MIT, SHA-256 | **IN opcional**, fuera del informe principal | `data/manifests/benchmark-decimer-smoke.fetch.json` |

## 3. Regla de `n` (estandarización)

- **`n` se estandariza por dossier y por conjunto de IDs congelado.** Nunca “todo a 1441” entre dossiers.
- **Dentro de un dossier:** todas las variantes comparadas ven los **mismos IDs** (comparación pareada). Congelar subconjunto de iteración con **seed + IDs en manifiesto**; el conjunto completo del dossier es el resultado final.
  - **Impreso:** n=1441 (CLEF+JPO), ya pareado por construcción.
  - **Manuscrito:** iterar sobre subconjunto congelado de IDs (HDM-500 fue una muestra *ad hoc*; reemplazar por subconjunto fijo/manifiesto para comparaciones formales); full 5088 para reporte final. Igualar a 1441 no aplica (no se compara con impreso).
  - **Clasificador:** cap estratificado por clase para iterar (ej. 10k, `[TBD]`), con seed/IDs; full split para reporte final.

## 4. Contratos compartidos (infraestructura)

### 4.1 Protocolo de medición pareado (por variante)
Cada variante se mide sobre el **mismo set congelado / mismos IDs** (piloto ⊂ eval ⊂ pool; nunca mezclar sets entre variantes). Por cada corrida registrar:
- **Precisión/tarea:** métrica definida por dossier (OCSR: idealmente exactitud SMILES **canónica con RDKit** `[pendiente integrar]` + `exact_match` cadena como dato; fidelidad `tf_match` ONNX↔TF). Reportar **Δ absoluto en pp contra FP32** (misma métrica, mismos IDs) además del valor absoluto.
- **Latencia:** **p50 y p95** en el **mismo hardware** y **mismo execution provider** (p. ej. ORT `CPUExecutionProvider`), registrados. No reportar solo media; la cola (p95) es relevante.
- **Tamaño en disco:** MB de los artefactos del modelo (encoder+decoder u equivalente), por variante.
- **Throughput:** si se procesan lotes, reportar lotes/s o imágenes/s con tamaño de lote `[si aplica]`.
- **Comparación de variantes (pareada):** **McNemar** sobre pares discordantes + Δ pp con **IC bootstrap**; reportar `n` + IC.
- **Caché:** predicciones por `(model_key, variant_key, image_id)` para no recalcular (`data/cache/predictions/`, gitignored).

**Advertencia de medición:** la cuantización dinámica (INT8) **no siempre es más rápida** (el overhead de cuantizar activaciones en runtime puede anular/peorar la ganancia). La latencia se mide, no se asume.

### 4.2 Etiquetas / métricas por bloque
- **OCSR (impreso/manuscrito):** `tf_match` (fidelidad ONNX↔TF, protocolo registrado), exactitud canónica RDKit `[TBD integrar]` + `exact_match` cadena, `valid`/`incomplete`, latencia p50/p95, tamaño, throughput si aplica.
- **Clasificador:** **matriz 4×4** (`one_molecule`, `several_molecules`, `other`, `reactions`) + binario colapsado visible (`one/several_molecule→structure`, `other→no-structure`) con **`reactions` excluida del binario** y reportada aparte (`ambiguous_excluded`). Fidelidad ONNX↔baseline si aplica.
- **Criterios numéricos:** placeholders `[TBD]` por dossier (pueden diferir). Criterios típicos a fijar: `Δ vs FP32 ≤ [TBD] pp` (precisión/fidelidad), p95 `≤ [TBD] ms`.

### 4.2 Preprocesado (registrado por modelo)
Registrar entrada, resize/normalización/canales, ruta TF baseline vs runtime ONNX usada en `tf_match`, versión. Control **FP32+TF** antes de concluir sobre cuantización (separa export/preprocess vs cuantización).

## 5. Reporte

- **Tres secciones independientes** (una por dossier), cada una con su `n`, IC y procedencia. Sin tabla comparativa única entre modelos, sin pooled entre dossiers, sin dominancia de HDM.
- Dentro de cada dossier: vista por fuente/subset si aplica + vista pareada (mismos IDs) entre variantes. Impreso puede incluir vista pooled **intra-dossier** etiquetada (CLEF+JPO, n=1441).
- `docs/benchmark-report.md` generado por `decimer-export report` desde `data/benchmark/**/*.json` y manifiestos. Alinear generador a este modelo de 3 dossiers (pendiente).

## 6. Orden de trabajo

1. **Specs por dossier** (este doc + 3 dossiers de una página) — hecho.
2. **Impreso:** control FP32+TF si falta; fijar criterio; cerrar variante (E1/E3) + política de calibración con RDKit.
3. **Manuscrito:** congelar subconjunto de iteración (seed/IDs en manifiesto); comparar variantes sobre él; full 5088 para reporte final.
4. **Clasificador:** export ONNX + wiring CLI + correr 13378718 (zip extraído local). 6670746 sigue OUT OF SCOPE.
5. **Reporte:** tres secciones independientes, `n`+IC, pareado/McNemar donde aplique.

## 7. Estado actual (contexto)
- Medido (OCSR): CLEF/JPO E1+TF y E3 (join TF), n=1441 pareado. HDM `--limit 500` FP32/E1/E3+TF (muestra ad hoc). **HDM E1+TF full 5088** puede estar en ejecución → `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json`.
- No medido e2e: clasificador (falta export ONNX + wiring).
- Pendiente: RDKit canónico, criterios `[TBD]`, subconjunto congelado HDM (manifiesto), FP32 controles donde falten, `report.py` a 3 secciones, McNemar/IC en tooling.
