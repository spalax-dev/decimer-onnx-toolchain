# Dossier — OCSR impreso (`ocsr-printed`)

> Dossier independiente. No se compara con manuscrito ni con clasificador.

## Propósito
Validar conversión DECIMER printed → ONNX: fidelidad vs TF y calidad SMILES en dominio impreso; seleccionar variante (FP32/E1/E3) y política de calibración.

## Suite
| Fuente | n | Dominio | Licencia/procedencia | Manifiesto/labels |
|---|---:|---|---|---|
| `hf-clef` | 992 | impreso | Keylab/OCSR-Benchmarks, HF mixed-see-sources; verificar CLEF 2012 | `data/manifests/benchmark-hf-clef.import.json` |
| `hf-jpo` | 449 | impreso | idem; verificar JPO/ChemInfty/OSRA | `data/manifests/benchmark-hf-jpo.import.json` |
| **Total dossier (intra-dossier)** | **1441** | impreso | — | `data/benchmark/hf-clef/labels.jsonl`, `data/benchmark/hf-jpo/labels.jsonl` |

## `n` / IDs
- **n=1441**, pareado por construcción. Todas las variantes se evalúan sobre los mismos 1441 IDs. Vista pooled intra-dossier (CLEF+JPO) permitida si se etiqueta; vista por fuente primaria.

## Métricas
- `tf_match` (ONNX==TF, fidelidad end-to-end según harness; contrato de preprocesado registrado).
- Exactitud SMILES **canónica (RDKit)** `[pendiente integrar]` + `exact_match` cadena como dato.
- `valid`, `incomplete` (heurística), latencia p50/p95 (CPU/máquina registrada).
- Comparación de variantes: **McNemar** (pares discordantes) + Δ pp con **IC bootstrap**; `n`+IC en reporte.

## Criterios de aceptación (placeholders `[TBD]`)
| Criterio | Umbral |
|---|---|
| `tf_match` fidelidad | `≥ [TBD]%` (estricto 100% o tolerancia definida) |
| Δ canónica vs TF/etiqueta | `≤ [TBD]` pp / Tanimoto `[TBD]` |
| `valid` / `incomplete` | `≥ [TBD]%` / `≤ [TBD]%` |
| Latencia p95 | `≤ [TBD] ms` (hardware fijo) |

## Variantes a comparar
FP32 (control fidelidad), **E1 dynamic INT8**, **E3 hybrid static-enc QDQ + dynamic-dec**. (E2/E4: ver `data/artifacts/selection/ocsr-printed.selection.json`.)

## Calibración (E3)
Política `[TBD]`: fuente, `n`, método fijos; evaluación en **holdout** (no in-sample). Evidencia actual: E3 calibrado con 2 fixtures mostró regresión fuerte de `tf_match` en CLEF/JPO (ver reportes/selección) — define nueva calibración antes de reconsiderar.

## Estado
- Pesos: `data/weights/ocsr-printed/DECIMER_model`. Tokenizer: `data/artifacts/tokenizer_SMILES.vocab.json`.
- Medido: E1+TF y E3 (join TF) en CLEF/JPO. Pendiente: control FP32+TF en golden, RDKit canónico, fijar `[TBD]`, recalibrar E3 en holdout si aplica.
- Reportes: `data/benchmark/hf-{clef,jpo}/report.*.json` (gitignored).
