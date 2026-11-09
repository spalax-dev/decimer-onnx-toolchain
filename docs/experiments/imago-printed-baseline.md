# Experimento: baseline externo Imago (OCSR impreso) — dato de referencia

> **Alcance:** experimento de curiosidad/comparación. **No forma parte del reporte oficial** (`docs/benchmark-report.md`) ni del pipeline de construcción de Imago. Se registra como métrica de referencia porque Imago es una herramienta OCSR clásica que, al igual que el plan para DECIMER printed/handdrawn, puede **incrustarse/distribuirse en software** (no es un modelo ML).

## Herramienta
- **Imago OCR** (EPAM). Reconocimiento OCSR rule-based/C++.
- Paquete usado: `epam.imago==2.0.0rc1` (PyPI, pre-release; Apache-2.0 en repo actual). Instalado ad-hoc en el venv (`uv pip install --pre`), no declarado en `pyproject.toml`.
- Flujo medido: imagen → `imago.Imago().load_image_from_file/recognize()` → `molecule` (MDL molfile) → **SMILES canónico RDKit** (`Chem.MolFromSmiles` + `MolToSmiles(canonical=True)`). Comparación con la **métrica primaria OCSR: canónica RDKit**.

## Benchmark (mismo protocolo que el dossier impreso)
- Dossier/modelo objetivo: `ocsr-printed` (DECIMER v2 impreso, golden CLEF+JPO).
- Conjunto congelado: `data/manifests/frozen/printed.frozen.json`, **split `eval`, n=1000**, seed `20260308` (mismos IDs que FP32/E1/TF).
- Sin baseline TF de paridad (no aplica a herramienta externa); caché `(ocsr-printed, imago-2.0.0rc1)`.
- Ejecución resiliente: `scripts/benchmark_imago_resilient.py` (subproceso por imagen, timeout/recuperación) por crashes nativos observados (`free(): invalid pointer`) tras 107/1000 en ejecución in-process. 39 errores/crashes aislados; 107 cache hits de la corrida inicial.
- Artefacto (gitignored): `data/benchmark/eval-1000/printed.imago.eval1000.json`.

## Resultados (eval-1000 impreso)

| Métrica | Imago 2.0.0rc1 |
|---|---:|
| **Canonical RDKit (primaria)** | **69/582 = 0.1186** (582/1000 comparables; ambos parsean) — **❌ < 70%** |
| Exact string (secundaria) | 50/1000 = 0.0500 |
| valid / incomplete | 608 / 392 |
| errores (crash/timeout/parse/no-mol) | 39 (de 893 medidos en runner resiliente) |
| Latencia (n=893 medidos) | p50 **347.4 ms**, p95 **962.8 ms** |

## Comparación (mismo eval impreso, métrica primaria canónica)

| Sistema | Canónica (primaria) | ≥70% | Exact string |
|---|---:|:---:|---:|
| **TF baseline (DECIMER impreso, referencia)** | **627/884 = 0.7093** | ✅ | 12/1000 (0.0120) |
| FP32 ONNX | 614/872 = 0.7041 | ✅ | 12/1000 (0.0120) |
| E1 dyn INT8 ONNX | 597/881 = 0.6776 | ❌ | 12/1000 (0.0120) |
| **Imago 2.0.0rc1 (externo)** | **69/582 = 0.1186** | ❌ | 50/1000 (0.0500) |

**Pareado (mismos 1000 IDs, acierto canónico vs label):** TF acierta solo **575**, Imago solo **17**, ambos 52, ninguno 356. Δ ≈ **+59.1 pp** a favor de TF (sobre comparables respectivos). En este conjunto, **DECIMER impreso (TF/FP32) es muy superior a Imago** por precisión canónica; la diferencia es grande y pareada (concluyente en este eval). Imago además produce muchas salidas no comparables/incompletas.

## Caveats
- Herramienta antigua/rule-based; rendimiento dependiente del tipo/limpieza de imagen. Estos resultados aplican a **este frozen eval impreso** (CLEF+JPO curado), no generalizan sin más.
- Solo 582/1000 salidas fueron comparables canónicamente (parseo mol/SMILES en ambos lados). La tasa de completitud/parseo es parte relevante del desempeño práctico.
- Imago mostró inestabilidad nativa en este entorno; las métricas usan el runner resiliente. Latencia medida incluye overhead de subproceso por imagen (no comparable 1:1 con inferencia in-process de ORT/TF).
- Dato de referencia/embeddability; **no** incluido en `docs/benchmark-report.md`.
