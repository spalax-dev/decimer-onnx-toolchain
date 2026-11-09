# Dossier — Clasificador de imagen (`image-classifier`)

> Dossier independiente. No se compara con OCSR impreso/manuscrito.

## Propósito
Validar conversión DECIMER Image Classifier → ONNX: fidelidad vs baseline (si aplica) y calidad de clasificación structure/no-structure; cerrar export ONNX + wiring + selección de variante.

## Suite
| Fuente | n | Licencia/procedencia | Estado |
|---|---:|---|---|
| Zenodo **13378718** (`dataset_for_image_classifier.zip`) | por split extraído (p. ej. `classified` 4×4000, `for_model/test` 1604) | registro indica MIT; origen imágenes mixto (verificar) | **IN** — requiere extracción local |
| Zenodo **6670746** | 126.6 GB | CC-BY-4.0 | **OUT OF SCOPE (formalizado)** — no descargar, no usar (archivo único, sin test separado expuesto) |

Importador: `decimer-export import-benchmark --source classifier-13378718 --root <zip_extraido> --split <classified|for_model/test|...>`.

## Mapeo de etiquetas (contrato, visible)
| Categoría original | Binario | En matriz binaria | Reporte |
|---|---|:---:|---|
| `one_molecule` | `structure` | Sí | 4×4 + binario |
| `several_molecules` | `structure` | Sí | 4×4 + binario |
| `other` | `no-structure` | Sí | 4×4 + binario |
| `reactions` | — | **No** | métrica separada `ambiguous_excluded` (no entra en TP/TN/FP/FN) |

Reportar **4×4** y **binario colapsado**; la exclusión de `reactions` es explícita.

## `n` / IDs (congelado)
- **Pool congelado objetivo n=10000**, **estratificado por las 4 clases, clases raras sobrerrepresentadas** (seed `20260308`, método determinístico: floor por clase + remanente ponderado por sqrt(capacidad); ver `data/manifests/frozen/classifier.frozen.spec.json`).
- **Estado:** `pending_data` hasta importar etiquetas 13378718 (`decimer-export import-benchmark --source classifier-13378718 --root <zip_extraido>`). Generar: `python -m decimer_onnx_export.frozen --classifier-labels <labels.jsonl> --out data/manifests/frozen/classifier.frozen.json`.
- **Eval n=1000**, **Piloto n=300** = primeras 300 del eval (orden congelado fijo), mismos IDs para todas las variantes. Resto del pool para análisis/reporte ampliado según se defina.
- Cache de predicciones: `(model, variant, image_id)`.

## Métricas (protocolo pareado)
- Matriz 4×4 + binaria (solo 3 categorías mapeadas), accuracy/precision/recall/F1 `[TBD]`. `reactions` aparte.
- Fidelidad ONNX↔baseline/TF si aplica `[TBD]`. Latencia **p50/p95** (mismo hardware, mismo provider), **tamaño MB**, **throughput** si lotes. No solo media.
- **Δ absoluto en pp vs FP32** (misma métrica, mismos IDs congelados) + valor absoluto. Comparación pareada: **McNemar** + IC bootstrap donde aplique; `n`+IC. Piloto ⊂ eval ⊂ pool congelado. Cache `(model,variant,image_id)`.
- Nota: INT8 dinámico puede no ser más rápido; medir latencia.

## Criterios de aceptación (placeholders `[TBD]`)
| Criterio | Umbral |
|---|---|
| Métrica binaria principal | `[TBD]` |
| Fidelidad export (si aplica) | `[TBD]` |
| Latencia p95 | `≤ [TBD] ms` |

## Variantes a comparar (orden de medición)
1. **FP32** (control)
2. **INT8 dynamic (all)**
3. **INT8 dynamic (sin atención / MLP selectivo)** `[pendiente generar]`
4. **FP16**

Registro por variante: `model_key=image-classifier`, `variant_key` estable. Pendiente export ONNX + wiring predict.

## Estado
- Pesos: fetch disponible (`--model image-classifier`, commit pin). **Export ONNX no realizado.** Wiring predict en CLI/harness pendiente.
- Harness lógico/mapeo implementado en `benchmark_run`/importador; falta ejecución e2e sobre 13378718.
- Bloqueante inmediato: exportar clasificador a ONNX y definir split/`n` de iteración (manifiesto).
