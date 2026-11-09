# Especificación de benchmark — DECIMER → ONNX

> Documento de diseño/protocolo. Define bloques de tarea, compatibilidad modelo×dataset, métricas, vistas de reporte y orden de trabajo. Los criterios de aceptación numéricos son **placeholders** hasta que se fijen explícitamente.

**Estado:** borrador formalizado (2026-11-08).  
**Fuentes de verdad operativas:** `docs/benchmark-report.md` (resultados neutrales auto-generados), `data/manifests/*.json` (procedencia/checksums/licencias), `data/artifacts/selection/*` (decisiones de selección, separadas del informe).

---

## 1. Reencuadre: bloques de tarea (no comparar entre tareas)

Los modelos **no** se tratan como un solo eje comparable por una misma métrica de accuracy.

| Bloque | Tarea | Modelos | Etiqueta / salida | Métricas primarias |
|---|---|---|---|---|
| **OCSR** | Imagen de estructura química → SMILES | `ocsr-printed`, `ocsr-handdrawn` | SMILES (canónico para evaluación de tarea, ver §4) | `tf_match` (fidelidad ONNX↔TF), exactitud/canónica de SMILES (con RDKit), `valid`, `incomplete`, latencia |
| **Clasificador** | Imagen → clase de contenido químico | `image-classifier` | 4 clases originales + colapso binario (ver §4) | Matriz 4×4, matriz binaria colapsada, métricas por clase, `reactions` reportada aparte; `tf_match` de fidelidad si aplica |

**Regla:** no se comparan scores de accuracy entre OCSR y clasificador. Dentro de OCSR, `ocsr-printed` y `ocsr-handdrawn` comparten tarea pero difieren en **dominio**; se evalúan en la matriz cross-domain (§2).

## 2. Matriz cross-domain OCSR (2×2)

|  | **CLEF+JPO** (dominio impreso) | **HDM** (dominio manuscrito) |
|---|---|---|
| **`ocsr-printed`** | in-domain | domain shift |
| **`ocsr-handdrawn`** | domain shift | in-domain |

- **Celdas in-domain:** evaluación primaria de cada modelo en su dominio.
- **Celdas domain shift:** válidas y reportadas explícitamente como *domain shift*; son hallazgos, no errores de protocolo. Responden: ¿el modelo especializado generaliza al otro dominio o solo gana en el suyo?
- **Comparación cross-model limpia:** solo sobre la **vista pareada** (§5), mismos IDs/casos evaluados por los modelos que puedan correr esa celda.

## 3. Data cards y matriz de compatibilidad

### 3.1 Data cards (fuentes)

| ID fuente | Bloque | Domain tag | n | Etiqueta | Test separado | Licencia / procedencia | Estado scope | Preprocesado | Manifiesto / labels |
|---|---|---|---:|---|---|---|---|---|---|
| `hf-clef` | OCSR | impreso | 992 | SMILES (`SMILES`) | split `test` (HF) | Keylab/OCSR-Benchmarks, commit obs. `2fdcdc59`; HF agregado **mixed-see-sources**; verificar términos originales CLEF 2012 antes de redistribuir | **IN** | contrato por modelo (§4) | `data/manifests/benchmark-hf-clef.import.json`, `data/benchmark/hf-clef/labels.jsonl` |
| `hf-jpo` | OCSR | impreso | 449 | SMILES | split `test` (HF) | idem agregado; verificar términos originales JPO/ChemInfty/OSRA | **IN** | contrato por modelo | `data/manifests/benchmark-hf-jpo.import.json`, `data/benchmark/hf-jpo/labels.jsonl` |
| `handdrawn-hdm` | OCSR | manuscrito | 5088 | SMILES (TSV `IDs\tSMILES`) | dataset completo usado como benchmark | Zenodo 7617107, DOI 10.5281/zenodo.7617107, **CC-BY-4.0**, md5 verificados | **IN** | contrato por modelo | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |
| `decimer-smoke` | OCSR | impreso (sanity) | 4 | SMILES | oficial DECIMER (no golden) | DECIMER-Image_Transformer `@9d06861`, MIT, SHA-256 pin | **IN (sanity/regresión opcional, fuera del informe principal)** | contrato por modelo | `data/manifests/benchmark-decimer-smoke.fetch.json`, `data/benchmark/decimer-smoke/labels.jsonl` |
| `classifier-13378718` | Clasificador | mixto (según categorías) | definido por split extraído | 4 clases: `one_molecule`, `several_molecules`, `other`, `reactions` | depende del split (`classified`, `for_model/test`, etc.) | Zenodo 13378718, md5 zip `e7e47f2a16735238104e8dd7ec5e65af`; registro indica MIT; origen de imágenes mixto (verificar) | **IN** (requiere extracción local + export ONNX clasificador) | contrato clasificador | importador `--source classifier-13378718`; manifiesto al importar |
| Zenodo **6670746** | Clasificador | — | 126.6 GB (archivo único) | structure/non-structure (dentro del tar) | no expuesto por separado en API | Zenodo 6670746, CC-BY-4.0, archivo único `DECIMER_Image_classifier_dataset.tar.gz`, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | **OUT OF SCOPE (formalizado)** — no se descarga ni usa en este benchmark por tamaño y ausencia de test separado expuesto | n/a | ninguno (decisión explícita) |

### 3.2 Matriz de compatibilidad modelo × dataset

| Modelo \ Dataset | `hf-clef` | `hf-jpo` | `handdrawn-hdm` | `classifier-13378718` |
|---|---|---|---|---|
| `ocsr-printed` | **válida, in-domain** | **válida, in-domain** | **válida, domain shift** | task-mismatch (no `exact_match` SMILES) |
| `ocsr-handdrawn` | **válida, domain shift** | **válida, domain shift** | **válida, in-domain** | task-mismatch |
| `image-classifier` | task-mismatch | task-mismatch | task-mismatch | **válida** (métricas de clasificación) |

**Reglas:** `task-mismatch` no se fuerza a la métrica del otro bloque. Las celdas `domain shift` se ejecutan y reportan con esa etiqueta.

## 4. Contratos comunes

### 4.1 Espacio de etiquetas
- **OCSR (tarea):** SMILES. Para exactitud química, comparar sobre **SMILES canónico (RDKit)** en el entorno de benchmark/export; mantener también `exact_match` de cadena como dato separado si se desea, pero no como única métrica química.
- **Clasificador:** reportar **4×4** sobre clases originales (`one_molecule`, `several_molecules`, `other`, `reactions`) **y** binario colapsado. Colapso binario definido: `one_molecule→structure`, `several_molecules→structure`, `other→no-structure`; **`reactions` excluida del binario** y reportada como métrica separada (`ambiguous_excluded`). La decisión de colapso/exclusión es visible (no oculta).

### 4.2 Contrato de preprocesado por modelo (registrado)
Registrar por modelo/variante: entrada esperada, resize/normalización/canales, ruta TF baseline vs ruta runtime ONNX usada en `tf_match`, y versión de implementación. Objetivo: atribuir divergencias de `tf_match` a **export/preprocess** vs **cuantización** (no conjeturar).

| Modelo | Preprocesado baseline TF (harness) | Preprocesado ONNX/runtime | Nota |
|---|---|---|---|
| `ocsr-printed` | `DECIMER.pre_process.decode_image` / ruta TF de referencia | `decimer_onnx.preprocess` (NumPy/Pillow) | documentar equivalencia/deltas; control FP32 requerido |
| `ocsr-handdrawn` | idem familia DECIMER, modelo hand-drawn | idem runtime | idem |
| `image-classifier` | según pipeline clasificador | según export | pendiente de formalizar al exportar |

## 5. Vistas de reporte (nunca pooled silencioso)

1. **Por fuente (primaria, siempre):** resultados por dataset/subset (`hf-clef`, `hf-jpo`, `handdrawn-hdm`, clasificador por split), con procedencia/licencia por subset.
2. **Pareada (cross-model limpia):** mismos IDs/casos evaluados por todos los modelos aplicables a la celda/tarea. Es la base para comparar `ocsr-printed` vs `ocsr-handdrawn` dentro de OCSR.
3. **Pooled solo intra-dominio, etiquetado:** permitido como vista adicional explícita (p. ej. **`ocsr-printed` pooled impreso = CLEF+JPO, n=1441**, tasas ponderadas por `n`; latencia global calculada sobre latencias combinadas, no promedio de percentiles). **Nunca** pooled printed+handdrawn en un solo número. Nunca reemplaza la vista por fuente.

**Mismo `n`:** dentro de una suite/celda, variantes comparadas sobre los mismos casos. Entre suites, usar `n` nativo por fuente o un **subconjunto pareado fijo** reproducible (IDs/seed en manifiesto) como vista adicional.

## 6. Métricas y criterios de aceptación (placeholders)

| Métrica | Aplica a | Placeholder criterio (fijar) |
|---|---|---|
| `tf_match` (fidelidad ONNX↔TF) | OCSR, clasificador si hay baseline | `≥ [TBD]%` por celda; estricto 100% para cuantización sin degradación aceptada |
| Δ exactitud canónica SMILES (vs TF o vs etiqueta canónica) | OCSR | `≤ [TBD]` pp / `[TBD]` Tanimoto medio |
| `exact_match` cadena (dato) | OCSR | reportado; no criterio químico único sin canonización |
| `valid` / `incomplete` | OCSR | `valid ≥ [TBD]%`, `incomplete ≤ [TBD]%` |
| Latencia `p95` | todos | `≤ [TBD] ms` en hardware fijo (§7) |
| Clasificación 4×4 / binaria / `reactions` separada | clasificador | `[TBD]` por métrica; `reactions` nunca dentro del binario |

## 7. Orden de trabajo (dependencias) y costo

1. **Especificación (este documento)** — fija protocolo antes de gastar cómputo. **Hecho**.
2. **Control FP32+TF** en celdas válidas (incluida 2×2 OCSR, mismo protocolo/IDs comparables) para separar gap **export/preprocess** vs **cuantización**. Independiente de variedad de datos.
3. **RDKit canónico** en env benchmark/export (decisión de tooling). Habilita métrica química.
4. **Criterios de aceptación** fijados con (2)+(3) → decidir E1/E3/otros por modelo/celda.
5. **Calibración E3** solo si aplica: fuente, `n`, método fijos; evaluación en **holdout** (no in-sample).
6. **Clasificador**: export ONNX + wiring CLI + métricas propias + 13378718 (zip extraído local). 6670746 permanece **OUT OF SCOPE**.
7. **Informe con 3 vistas** + matriz cross-domain, regenerable (`decimer-export report`).

**Costo operativo:** cachear salidas TF por `(modelo, dataset)`; iterar con **subconjunto pareado fijo** (IDs/seed en manifiesto); reservar full para reporte final; fijar/registrar hardware, hilos y versiones.

## 8. Licencias al mezclar fuentes
- Mantener procedencia/licencia **por subset** en manifiestos/reportes (ya aplicado). HF `mixed-see-sources`: verificar términos originales por subset **antes de redistribuir** imágenes/artefactos derivados.
- 6670746: fuera de scope; no descargar, no redistribuir.

## 9. Estado actual (contexto, no parte del protocolo)
- Implementado: fetch pesos/benchmarks, import HF CLEF/JPO, HDM, harness OCSR (`tf_match`/`exact`/`valid`/latencia), importador clasificador 13378718 con mapeo, generador `docs/benchmark-report.md`.
- Medido: CLEF/JPO (E1, E3 con join TF); HDM `--limit 500` (FP32, E1, E3). **HDM E1+TF completo 5088** puede estar en ejecución → `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json`.
- Pendiente de protocolo: control FP32 en golden, RDKit, umbrales, calibración E3 holdout, clasificador ONNX e2e, vistas pareada/pooled/matriz en el informe, `report.py` alineado a §5.
