# INF-8239 · Unidad 03 · Ejercicio 05 · Motor recomendador reproducible

**Estudiante:** Johanna Joaquín · **Asignatura:** INF-8239 Ciencia de Datos II · Universidad Autónoma de Santo Domingo
**Código del ejercicio:** U03.E05 · **Base del proyecto:** material de Edwin Ramón José Nolasco (LAB08 y LAB09)

## Resumen

Se comparan cuatro familias de recomendadores sobre MovieLens `ml-latest-small` con la misma partición temporal (la última valoración de cada usuario se reserva para evaluación): popularidad, contenido (géneros con TF-IDF), factorización matricial (MF) e híbrido (re-ordena los 100 mejores candidatos de MF mezclando la señal colaborativa y la de contenido con el peso `alpha`).

**Decisión final.** Usar el **híbrido con 20 factores y α = 0.75** para usuarios con historial, y **popularidad suavizada como fallback no personalizado** para usuarios nuevos.

- Su HitRate@10 (0.039) **no se distingue estadísticamente** del baseline de popularidad por conteo (0.043; error estándar ≈ 0.008). No se afirma que acierte más.
- Su ventaja es la **cobertura del catálogo** (7.6 % frente a 1.2 %) y menor concentración en las 100 películas más populares (57 % de las posiciones frente a 99 %).
- Su costo es mayor: unos 10 s de entrenamiento y 1.6 MB frente a 76 KB.
- Si el único criterio fuera aciertos por costo, el baseline de popularidad por conteo sería la elección.

## Resultados principales

Evaluación sobre 587 usuarios, media de 3 semillas (42, 43, 44) para los modelos entrenados. Popularidad y contenido son deterministas.

| Configuración | HitRate@10 | NDCG@10 | Cobertura | RMSE | Entrenamiento (s) | Tamaño (KB) |
|---|---|---|---|---|---|---|
| Popularidad por conteo (baseline) | 0.0426 | 0.0197 | 0.012 | n/a | 0.0 | 76 |
| Popularidad suavizada | 0.0324 | 0.0168 | 0.006 | n/a | 0.0 | 15 |
| Contenido (géneros, desempate por popularidad) | 0.0051 | 0.0021 | 0.082 | n/a | 0.0 | 1819 |
| Factorización MF, 20 factores | 0.0386 ± 0.0020 | 0.0185 | 0.074 | 1.025 | 9.4 | 1611 |
| **Híbrido, 20 factores, α = 0.75** | **0.0392 ± 0.0017** | 0.0184 | 0.076 | 1.025 | 9.4 | 1611 |
| Híbrido, 40 factores, α = 0.25 | 0.0256 ± 0.0030 | 0.0124 | 0.122 | 1.019 | 9.3 | 3222 |

Referencias de RMSE: predecir siempre la media global da 1.116 y la media por película da 1.050. La tabla completa (15 configuraciones, con Pareto) está en `reports/e05_tabla_principal.md`.

## Cómo reproducir

Requisitos: Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.12
uv sync
uv run pytest -q
uv run python scripts/download_data.py      # descarga MovieLens (no se sube al repositorio)
uv run python scripts/audit_data.py         # hash, conteos y validaciones
uv run python scripts/lab08_content.py      # popularidad suavizada y contenido
uv run python scripts/e05_experimentos.py   # comparación completa (unos 95 s)
```

En Windows PowerShell, antes del experimento: `$env:PYTHONUTF8=1`.

Notebook de resultados (lee `reports/`):

```bash
uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace notebooks/E05_resultados.ipynb
```

La aplicación local del laboratorio: `uv run streamlit run app/streamlit_app.py`.

## Estructura

```
data/raw/           MovieLens descargado (ignorado por Git; no se redistribuye)
docs/               DATASET_CARD.md, SYSTEM_CARD.md, USO_IA.md
notebooks/          E05_resultados.ipynb (ejecutado)
reports/            métricas, tablas, perfiles y figura de Pareto
scripts/            descarga, auditoría y experimentos (lab08, lab09, e05)
src/inf8239_u03/    datos, modelos, métricas y evaluación
tests/              pruebas de datos, modelos, métricas, evaluación y corte temporal
```

## Datos, licencia y términos de uso

MovieLens Latest Small (GroupLens Research, Universidad de Minnesota): 100,836 valoraciones de 610 usuarios sobre 9,724 películas. Se usa con fines académicos. **No se redistribuye:** `data/raw/` está en `.gitignore` y el repositorio solo contiene el script de descarga. Detalles, hash y limitaciones en `docs/DATASET_CARD.md`.

## Partición y evaluación

- **Corte temporal:** la última valoración de cada usuario va a evaluación; el resto, a entrenamiento (100,226 valoraciones). Una partición aleatoria permitiría aprender de interacciones posteriores.
- **Usuarios evaluables:** 587 de 610 (en 23 usuarios la película retenida no aparece en entrenamiento).
- **Métricas:** HitRate@10, NDCG@10, cobertura del catálogo, concentración en películas populares, RMSE (solo modelos de ratings), tiempo y tamaño del modelo.
- **Usuarios fríos:** segmentación por tamaño de historial (Q1 ≤ 34 valoraciones, Q3 ≥ 161) y perfil de usuario nuevo con fallback declarado como no personalizado.
- **Pareto:** frontera con 3 objetivos (HitRate, cobertura, costo) y con 2 objetivos (HitRate contra costo). Con 2 objetivos solo quedan las dos popularidades.

## Errores, límites y riesgos

- Una sola valoración retenida por usuario produce HitRate muy ruidoso (cada punto equivale a unos 6 usuarios): las diferencias entre popularidad, MF e híbrido no son concluyentes.
- La MF es un descenso de gradiente simple, sin sesgos por usuario ni película y sin ajuste de hiperparámetros.
- Sesgo de selección: solo se observan valoraciones de películas que las personas decidieron ver. No hay demografía, así que no se pueden afirmar resultados por grupos.
- Riesgos del sistema: popularidad dominante, burbuja de filtro y retroalimentación. Ver `docs/SYSTEM_CARD.md`.
- Un buen resultado en MovieLens no demuestra desempeño en otro catálogo o población.

## Pruebas

`uv run pytest -q` ejecuta las pruebas del proyecto base (datos, contenido, factorización y métricas) y las del ejercicio (`test_evaluation.py` y `test_temporal_split.py`).

## Uso de inteligencia artificial

Declarado en `docs/USO_IA.md`: herramienta, propósito, verificaciones y correcciones.

## Referencia

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), 1–19. https://doi.org/10.1145/2827872
