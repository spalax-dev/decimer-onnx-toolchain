# Dossier — OCSR manuscrito (`ocsr-handdrawn`)

> Dossier independiente. No se compara con impreso ni con clasificador.

## Propósito
Validar conversión DECIMER hand-drawn → ONNX: fidelidad vs TF y calidad SMILES en dominio manuscrito; seleccionar variante (FP32/E1/E3) y política de calibración.

## Suite
| Fuente | n | Dominio | Licencia/procedencia | Manifiesto/labels |
|---|---:|---|---|---|
| `handdrawn-hdm` (Zenodo 7617107) | 5088 | manuscrito | CC-BY-4.0, md5 verificados | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |

Estratificar por subfuente si es identificable desde IDs/estructura original.

## `n` / IDs (congelado)
- **Pool congelado n=1400** (seed `20260308`, selección determinística desde 5088 HDM) → `data/manifests/frozen/handdrawn.frozen.json`.
- **Eval n=1000** = primeras 1000 del pool congelado (orden fijo). **Piloto n=300** = primeras 300 del eval (mismos IDs para todas las variantes).
- **Full dossier n=5088** reservado para reporte final. `--limit 500` previo = muestra ad hoc (no usar como base de decisión formal). Todas las variantes comparadas sobre los mismos IDs del split (pareado).
- Cache de predicciones: `(model, variant, image_id)` en `data/cache/predictions/` (gitignored).

## Métricas (protocolo pareado)
- `tf_match` (ONNX==TF, fidelidad). Exactitud SMILES **canónica (RDKit)** `[pendiente integrar]` + `exact_match` cadena. `valid`, `incomplete`.
- Latencia **p50/p95** (mismo hardware, ORT `CPUExecutionProvider`), **tamaño MB**, **throughput** si lotes. No solo media.
- **Δ absoluto en pp vs FP32** (misma métrica, mismos IDs) + valor absoluto. **McNemar** + Δ pp con **IC bootstrap**; `n`+IC. Piloto ⊂ eval ⊂ pool congelado; muestras ad hoc no se usan para decisión formal. Cache `(model,variant,image_id)`.
- Nota: INT8 dinámico puede no ser más rápido; medir latencia.

## Criterios de aceptación (placeholders `[TBD]`)
| Criterio | Umbral |
|---|---|
| `tf_match` fidelidad | `≥ [TBD]%` |
| Δ canónica vs TF/etiqueta | `≤ [TBD]` pp / Tanimoto `[TBD]` |
| `valid` / `incomplete` | `≥ [TBD]%` / `≤ [TBD]%` |
| Latencia p95 | `≤ [TBD] ms` (hardware fijo) |

## Variantes a comparar (orden de medición)
1. **FP32** (control fidelidad)
2. **INT8 dynamic (all)** — (actual: artefactos E1)
3. **INT8 dynamic (sin atención)** `[pendiente generar]`
4. **FP16** `[pendiente; evaluar viabilidad ORT CPU/target]`
5. **INT8 estática (si 2–4 fallan criterios)** — holdout `[condicional]`

Registro por variante: `model_key=ocsr-handdrawn`, `variant_key` estable, artefactos y config.

## Calibración (E3)
Política `[TBD]`: fuente, `n`, método fijos; evaluación en **holdout**. Evidencia actual: E3 calibrado con 32 HDM mostró `tf_match` 0/500 y 203 incompletos en muestra ad hoc — recalibrar/evaluar bajo protocolo holdout antes de concluir.

## Estado
- Pesos: `data/weights/ocsr-handdrawn/DECIMER_HandDrawn_model`. Tokenizer: `data/artifacts/tokenizer_handdrawn.vocab.json` (`token_to_id` idéntico a printed).
- Medido: FP32/E1/E3+TF `--limit 500` (muestra ad hoc). **E1+TF full 5088** puede estar en ejecución → `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json`.
- Pendiente: congelar IDs de iteración (manifiesto), FP32 control en subconjunto/full según costo, RDKit, fijar `[TBD]`, E3 holdout.
