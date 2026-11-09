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

## `n` / IDs
- **Iteración:** cap estratificado por clase, seed + IDs en manifiesto (p. ej. `[TBD]`, hasta 10k), para iterar rápido.
- **Reporte final:** split completo definido (p. ej. `for_model/test` o el acordado), documentado.

## Métricas
- Matriz 4×4, matriz binaria (solo 3 categorías mapeadas), accuracy/precision/recall/F1 según se defina `[TBD]`.
- `reactions`: conteo y distribución de predicciones, reportada aparte.
- Fidelidad ONNX↔baseline/TF si se define baseline comparable `[TBD]`.
- Latencia p50/p95 (hardware fijo).

## Criterios de aceptación (placeholders `[TBD]`)
| Criterio | Umbral |
|---|---|
| Métrica binaria principal | `[TBD]` |
| Fidelidad export (si aplica) | `[TBD]` |
| Latencia p95 | `≤ [TBD] ms` |

## Variantes a comparar
FP32 + variantes de cuantización `[TBD]` tras exportar clasificador.

## Estado
- Pesos: fetch disponible (`--model image-classifier`, commit pin). **Export ONNX no realizado.** Wiring predict en CLI/harness pendiente.
- Harness lógico/mapeo implementado en `benchmark_run`/importador; falta ejecución e2e sobre 13378718.
- Bloqueante inmediato: exportar clasificador a ONNX y definir split/`n` de iteración (manifiesto).
