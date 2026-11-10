# Matriz de calificación y puntaje por variante (decisión de artefactos)

> Documento de **síntesis argumentada** de la selección para los **tres modelos distintos** del proyecto (`ocsr-printed`, `ocsr-handdrawn`, `image-classifier`), evaluados en dossiers independientes. Complementa (no reemplaza) la evidencia cruda en `docs/benchmark-report.md` y los registros en `data/artifacts/selection/*.selection.json`. Las ponderaciones reflejan las prioridades del proyecto; los puntajes por criterio se derivan de resultados medidos con reglas explícitas.

## 1. Metodología

- **Puntaje final:** escala **0–100** por variante, calculado como suma ponderada: `Σ (peso_i × puntaje_i) / 100`, donde cada criterio se puntúa **0–100** y los pesos suman **100%** dentro de cada dossier.
- **Fuente de evidencia:** splits congelados (seed `20260308`, mismos IDs), métrica primaria del dossier, reportes `data/benchmark/{eval-1000,pilot,remainder}/*.json`. Latencias p50 en ms (CPU/ORT); cuando hubo contención se prioriza la comparación cualitativa/mediciones en solitario (la accuracy no se ve afectada).
- **Regla de piso (gate):** variantes con métrica primaria **<70%** en `eval` congelado no son aceptables. En OCSR además se exige **robustez** (no basta un borde en eval si cae en `remainder`/held-out). Un fallo de piso/robustez fuerza el puntaje de “Calidad/precisión” a 0–30 y limita la elegibilidad, aunque el total ponderado sea alto en otros ejes.

### Normalización usada (criterio → puntaje 0–100)
| Criterio | Cómo se puntúa (0–100) |
|---|---|
| **Calidad/precisión primaria** (gate, **determinante**) | ≥75%: 90–100; 70–74.9%: 70–89 (según margen y estabilidad); <70% o no robusto (borde eval pero <70% held-out): 0–40. 100 = mejor observado en el dossier manteniendo robustez. **Este criterio tiene peso determinante**: no cumplir el piso/robustez invalida la variante para selección. |
| **Robustez / generalización** | 100: cumple piso en eval y se sostiene (sin caída relevante en held-out cuando aplica); 50: borde sin evidencia robusta; 0–30: cae bajo piso o colapsa. |
| **Fidelidad (TF/paridad)** | Escala lineal aprox. sobre la métrica de fidelidad del dossier (`tf_match` OCSR, paridad TF clasificador): 100≈mejor/ref, 0≈fallo severo. |
| **Latencia** | 100 = más rápida (p50) entre opciones viables del dossier; escala relativa (mejor = 100, peor ≈ 0–30). Penaliza variantes más lentas que la referencia. |
| **Tamaño / compresión** | 100 = menor tamaño (mayor compresión) útil; 0 = mayor tamaño; ponderado según objetivo del dossier. |
| **Viabilidad runtime** | 100 = ejecuta correctamente en CPU/ORT; 0 = no viable (error runtime). |

> **Nota:** los puntajes son una **rúbrica transparente** para comparar trade-offs medidos, no una métrica “oficial” del modelo. El criterio dominante de aceptación es el piso ≥70% + robustez, reflejado con un **peso determinante** en accuracy/calidad.

---

## 2. Dossier `ocsr-printed` (impreso)

**Objetivo:** cumplir ≥70% canónico robusto; comprimir solo si no se degrada el piso.

| Criterio | Peso | Justificación del peso |
|---|---:|---|
| **Calidad/precisión canónica (gate, determinante)** | **60%** | Métrica primaria y piso obligatorio; factor determinante de elegibilidad. |
| Robustez (eval→held-out) | 15% | Evita seleccionar variantes borde/no generalizables. |
| Fidelidad TF (`tf_match`) | 10% | Preservar comportamiento del baseline. |
| Latencia (p50) | 5% | Relevante, pero secundaria frente a corrección. |
| Tamaño (compresión) | 5% | Deseable si no rompe el piso. |
| Viabilidad runtime | 5% | Debe ejecutarse en CPU/ORT. |

| Variante | Tamaño | Eval canónica | Held-out canónica | `tf_match` (eval) | p50 (ms) | Calidad (60) | Robustez (15) | Fidelidad (10) | Latencia (5) | Tamaño (5) | Viab. (5) | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FP32 (seleccionada)** | 301.2 MB | **0.7041** | — (ref. estable) | 0.820 | 2317 | 78 | 90 | 90 | 80 | 20 | 100 | **79.3** |
| FP32enc+e1sdec | 264.2 MB | 0.7000 (borde) | **0.6870** ❌ | 0.812 | 2813 | 30 | 30 | 88 | 65 | 35 | 100 | **41.3** |
| selCenc+e1sdec | 123.0 MB | 0.703* (piloto) | **0.6944** ❌ | ~0.797* | 3080* | 28 | 30 | 85 | 60 | 75 | 100 | **41.6** |
| selMidenc+e1sdec | 240.1 MB | 0.701* (piloto) | **0.6880** ❌ | ~0.803* | 2325* | 28 | 30 | 86 | 78 | 45 | 100 | **41.1** |
| E1 dynamic INT8 all | 81.3 MB | **0.6776** ❌ | — | 0.757 | 3816 | 20 | 40 | 70 | 45 | 90 | 100 | **36.8** |
| E3 hybrid (static enc) | 84.4 MB | **0.4854** ❌ | — | 0.389 | 3060 | 0 | 0 | 20 | 55 | 88 | 100 | **14.2** |
| E4 FP16 (OCSR) | — | — | — | — | — | 0 | 0 | 0 | 0 | 60 | 0 | **3.0** |

\* señal de piloto (n=300), usada como screening; la decisión de robustez se toma con `remainder`/eval.

**Argumento:** con **accuracy como criterio determinante (60%)**, FP32 es la única variante que supera el gate de **≥70% canónico de forma robusta**. Las alternativas comprimidas/mixtas o fallan el piso en eval (E1/E3) o son borde y **no se sostienen en held-out** (caen ~1–3 pp bajo 70%), por lo que no son elegibles pese a mejor tamaño/compresión. E4 FP16 no es viable en ORT CPU. → **Selección: FP32**.

---

## 3. Dossier `ocsr-handdrawn` (manuscrito)

**Objetivo:** cumplir ≥70% canónico; entre opciones que cumplen, priorizar calidad/fidelidad y latencia para uso por defecto.

| Criterio | Peso | Justificación del peso |
|---|---:|---|
| **Calidad/precisión canónica (gate, determinante)** | **60%** | Métrica primaria y piso obligatorio; factor determinante de elegibilidad. |
| Fidelidad TF (`tf_match`) | 15% | Importante para preservar comportamiento. |
| Latencia (p50) | 10% | Relevante para experiencia de uso. |
| Robustez | 5% | Estabilidad de la variante. |
| Tamaño (compresión) | 5% | Deseable, secundario si hay trade-off fuerte. |
| Viabilidad runtime | 5% | Debe ejecutarse en CPU/ORT. |

| Variante | Tamaño | Eval canónica | `tf_match` | p50 / p95 (ms) | Calidad (60) | Fidelidad (15) | Latencia (10) | Robustez (5) | Tamaño (5) | Viab. (5) | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FP32 (seleccionada)** | 301.2 MB | **0.7300** | **0.912** | **1115 / 2156** | 88 | 100 | 100 | 90 | 20 | 100 | **88.3** |
| E1 dynamic INT8 all | 81.3 MB | **0.7149** (−1.51 pp) | **0.858** (−5.4 pp) | **2860 / 3951** (~2.6×) | 78 | 82 | 35 | 85 | 90 | 100 | **76.4** |
| E3 hybrid (static enc) | 84.4 MB | **0.0000** ❌ | 0.000 | 17705 / 27366 | 0 | 0 | 0 | 0 | 88 | 100 | **9.4** |

**Argumento:** con **accuracy como criterio determinante (60%)**, E1 **sí cumple** el piso (≥70%) pero queda por debajo de FP32 en canónica (−1.51 pp) y además degrada fidelidad y latencia (~2.6× más lento; la cuantización dinámica no garantiza aceleración en CPU). **FP32** ofrece mayor calidad, fidelidad y latencia con holgura sobre el piso, por lo que es la opción por defecto para `chemistry_draw`; E1 queda como alternativa válida registrada si se prioriza tamaño sobre velocidad/fidelidad. → **Selección: FP32**.

---

## 4. Dossier `image-classifier` (clasificador)

**Objetivo:** cumplir ≥70% accuracy binaria y **mejorar compresión** manteniendo fidelidad.

| Criterio | Peso | Justificación del peso |
|---|---:|---|
| **Calidad (accuracy binaria, gate, determinante)** | **55%** | Métrica primaria y piso obligatorio; factor determinante de elegibilidad. |
| Tamaño / compresión | 25% | Objetivo explícito (reducir tamaño). |
| Fidelidad/paridad TF | 10% | Preservar comportamiento del baseline. |
| Latencia | 5% | Relevante, sin degradar. |
| Viabilidad runtime | 5% | Debe ejecutarse correctamente. |

| Variante | Tamaño | Accuracy binaria (eval) | Paridad TF | Latencia | Calidad (55) | Tamaño (25) | Fidelidad (10) | Latencia (5) | Viab. (5) | **Total** |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **E4 FP16 (seleccionada)** | **7.8 MB** (−49%) | **0.7679** | **0.998** | ~≈ FP32 CPU | 90 | 100 | 99 | 85 | 100 | **93.7** |
| FP32 (referencia) | 15.4 MB | 0.7666 | 1.000 (ref) | ref | 90 | 40 | 100 | 85 | 100 | **78.8** |
| E1s selective (no Conv) | ~15.4 MB | 0.7666 | alta | — | 90 | 40 | 98 | 85 | 100 | **78.6** |
| E2 static QDQ (holdout) | 4.8 MB | **0.5345** ❌ | — | — | 10 | 100 | 40 | 90 | 100 | **44.0** |
| E1 dynamic INT8 all | 4.3 MB | **0.4085** ❌ | — | — | 0 | 100 | 20 | 90 | 100 | **36.5** |

**Argumento:** con **accuracy como criterio determinante (55%)**, E4 FP16 es la **única** variante que cumple ≥70% y a la vez logra compresión sustancial (−49%) con paridad TF muy alta, sin penalizar la accuracy. E1-all/E2 caen muy por debajo del piso por cuantizar Conv agresivamente; E1s conserva accuracy pero no comprime. → **Selección: E4 FP16**.

---

## 5. Resumen de selección

| Dossier | Seleccionada | Puntaje | Motivo principal |
|---|---|---:|---|
| `ocsr-printed` | **FP32** | **79.3** | Única variante robustamente ≥70% canónico (accuracy determinante); alternativas comprimidas no sostienen el piso en held-out. |
| `ocsr-handdrawn` | **FP32** | **88.3** | Mejor calidad/fidelidad/latencia entre las que cumplen ≥70%; E1 válido en accuracy pero con trade-offs. |
| `image-classifier` | **E4 FP16** | **93.7** | Cumple ≥70% (determinante) y maximiza compresión con alta fidelidad (mejor balance objetivo). |

*Los puntajes sintetizan evidencia medida bajo el protocolo descrito; la **accuracy tiene peso determinante** (gate ≥70% + robustez) y cualquier cambio de pesos/criterios desplazaría el ranking de forma explícita en la tabla.*
