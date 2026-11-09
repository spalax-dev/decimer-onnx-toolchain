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

## `n` / IDs (congelado)
- **Pool congelado n=1400** (seed `20260308`, selección determinística desde 1441 disponibles; 41 excluidos registrados por método) → `data/manifests/frozen/printed.frozen.json`.
- **Eval n=1000** = primeras 1000 del pool congelado (orden fijo). **Piloto n=300** = primeras 300 del eval (mismos IDs para todas las variantes).
- Todas las variantes comparadas sobre los **mismos IDs** del split (pareado). Vista por fuente primaria + pooled intra-dossier etiquetado (CLEF+JPO dentro del split).
- Cache de predicciones: `(model, variant, image_id)` en `data/cache/predictions/` (gitignored), reutilizable vía `--model-key/--variant-key/--cache-dir`.

## Métricas (protocolo pareado)
- `tf_match` (ONNX==TF, fidelidad end-to-end; contrato de preprocesado registrado). Exactitud SMILES **canónica (RDKit)** `[pendiente integrar]` + `exact_match` cadena como dato. `valid`, `incomplete`.
- Latencia **p50/p95** (mismo hardware, ORT `CPUExecutionProvider`), **tamaño MB** (encoder+decoder), **throughput** si lotes. No solo media.
- Reportar **Δ absoluto en pp vs FP32** (misma métrica, mismos IDs) + valor absoluto. Comparación pareada: **McNemar** + Δ pp con **IC bootstrap**; `n`+IC. Piloto ⊂ eval ⊂ pool congelado. Cache `(model,variant,image_id)`.
- Nota: INT8 dinámico puede no ser más rápido (overhead de activaciones); medir latencia.

## Criterios de aceptación (placeholders `[TBD]`)
| Criterio | Umbral |
|---|---|
| `tf_match` fidelidad | `≥ [TBD]%` (estricto 100% o tolerancia definida) |
| Δ canónica vs TF/etiqueta | `≤ [TBD]` pp / Tanimoto `[TBD]` |
| `valid` / `incomplete` | `≥ [TBD]%` / `≤ [TBD]%` |
| Latencia p95 | `≤ [TBD] ms` (hardware fijo) |

## Variantes a comparar (orden de medición)
1. **FP32** (control fidelidad)
2. **INT8 dynamic (all)** — pesos/ops cuantizados dinámicamente (actual: artefactos E1)
3. **INT8 dynamic (sin atención)** — variante dinámica excluyendo atención `[pendiente generar/exportar si aplica]`
4. **FP16** `[pendiente: E4 decoder presentó error runtime en ORT CPU; evaluar viabilidad/runtime target]`
5. **INT8 estática (si 2–4 fallan criterios)** — calibración fija/holdout, método documentado `[condicional]`

Registro por variante: `model_key=ocsr-printed`, `variant_key` estable (ej. `fp32`, `int8-dyn-all`, `int8-dyn-noattn`, `fp16`, `int8-static`), artefactos y config.

## Calibración (E3)
Política `[TBD]`: fuente, `n`, método fijos; evaluación en **holdout** (no in-sample). Evidencia actual: E3 calibrado con 2 fixtures mostró regresión fuerte de `tf_match` en CLEF/JPO (ver reportes/selección) — define nueva calibración antes de reconsiderar.

## Estado
- Pesos: `data/weights/ocsr-printed/DECIMER_model`. Tokenizer: `data/artifacts/tokenizer_SMILES.vocab.json`.
- Medido: E1+TF y E3 (join TF) en CLEF/JPO. Pendiente: control FP32+TF en golden, RDKit canónico, fijar `[TBD]`, recalibrar E3 en holdout si aplica.
- Reportes: `data/benchmark/hf-{clef,jpo}/report.*.json` (gitignored).
