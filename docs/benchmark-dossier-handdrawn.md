# Dossier — OCSR manuscrito (`ocsr-handdrawn`)

> Dossier independiente. No se compara con impreso ni con clasificador.

## Propósito
Validar conversión DECIMER hand-drawn → ONNX: fidelidad vs TF y calidad SMILES en dominio manuscrito; seleccionar variante (FP32/E1/E3) y política de calibración.

## Suite
| Fuente | n | Dominio | Licencia/procedencia | Manifiesto/labels |
|---|---:|---|---|---|
| `handdrawn-hdm` (Zenodo 7617107) | 5088 | manuscrito | CC-BY-4.0, md5 verificados | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |

Estratificar por subfuente si es identificable desde IDs/estructura original.

## `n` / IDs
- **Full dossier n=5088** (reporte final).
- **Iteración:** subconjunto **congelado** de IDs (seed + IDs en manifiesto) para comparar variantes a bajo costo. `--limit 500` usado hasta ahora es muestra *ad hoc*; reemplazar por subconjunto fijo/manifiesto para comparaciones formales. No igualar a 1441 (no se compara con impreso).

## Métricas
- `tf_match` (ONNX==TF, fidelidad end-to-end; contrato de preprocesado registrado).
- Exactitud SMILES **canónica (RDKit)** `[pendiente integrar]` + `exact_match` cadena como dato.
- `valid`, `incomplete` (heurística), latencia p50/p95.
- Comparación de variantes (mismos IDs): **McNemar** + Δ pp con **IC bootstrap**; `n`+IC. Muestras parciales etiquetadas explícitamente, no mezcladas con full sin marcar `n`.

## Criterios de aceptación (placeholders `[TBD]`)
| Criterio | Umbral |
|---|---|
| `tf_match` fidelidad | `≥ [TBD]%` |
| Δ canónica vs TF/etiqueta | `≤ [TBD]` pp / Tanimoto `[TBD]` |
| `valid` / `incomplete` | `≥ [TBD]%` / `≤ [TBD]%` |
| Latencia p95 | `≤ [TBD] ms` (hardware fijo) |

## Variantes a comparar
FP32 (control), **E1 dynamic INT8**, **E3 hybrid static-enc QDQ + dynamic-dec**.

## Calibración (E3)
Política `[TBD]`: fuente, `n`, método fijos; evaluación en **holdout**. Evidencia actual: E3 calibrado con 32 HDM mostró `tf_match` 0/500 y 203 incompletos en muestra ad hoc — recalibrar/evaluar bajo protocolo holdout antes de concluir.

## Estado
- Pesos: `data/weights/ocsr-handdrawn/DECIMER_HandDrawn_model`. Tokenizer: `data/artifacts/tokenizer_handdrawn.vocab.json` (`token_to_id` idéntico a printed).
- Medido: FP32/E1/E3+TF `--limit 500` (muestra ad hoc). **E1+TF full 5088** puede estar en ejecución → `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json`.
- Pendiente: congelar IDs de iteración (manifiesto), FP32 control en subconjunto/full según costo, RDKit, fijar `[TBD]`, E3 holdout.
