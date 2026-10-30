# Declaración de uso de inteligencia artificial · U03.E05

**Estudiante:** Johanna Joaquín. **Herramienta:** Claude (Anthropic), asistente de IA, usado en esta conversación entre el 29 de octubre de 2026 y la entrega.

Yo respondo por el código, los resultados, las referencias y las conclusiones. Todo lo entregado lo ejecuté y puedo explicarlo.

| # | Propósito | Fragmento afectado | Prompt relevante (resumen) | Verificación realizada | Cambios y correcciones |
|---|---|---|---|---|---|
| 1 | Plan de trabajo a partir de LAB08, LAB09 y la rúbrica | Ruta de ejecución y estructura del informe | Pegué el manual de LAB08/LAB09 y el mandato del Ejercicio 05 y pedí los pasos | Contraste de cada paso con el manual y la rúbrica | Adapté los pasos a mi carpeta y comandos de PowerShell |
| 2 | Depuración de `scripts/lab09_hybrid.py` | Línea 45 (`profile = ...`) | Pegué el error `TypeError: np.matrix is not supported` | Reejecuté el script y obtuve métricas (HitRate 0.0375 con α=0.75; 0.0324 con α=0.25) | Corrección con `np.asarray`; un primer intento quedó con sangría errónea y lo corregí |
| 3 | Script de comparación de modelos | `scripts/e05_experimentos.py` | Pedí comparar popularidad, contenido, colaborativo e híbrido con tres semillas y Pareto | La IA lo comparó contra la lógica de `lab09_hybrid.py` en datos sintéticos; yo lo ejecuté con MovieLens real y revisé que los resultados fueran idénticos en dos corridas | Corrección de un error de impresión por codificación (`α` en Windows) y adición del Pareto de 2 objetivos |
| 4 | Funciones de evaluación y pruebas | `src/inf8239_u03/evaluation.py`, `tests/test_evaluation.py`, `tests/test_temporal_split.py` | Pedí métricas adicionales y pruebas | `uv run pytest -q` en verde | — |
| 5 | Redacción | README, Dataset Card, System Card e informe | Pedí borradores con mis resultados reales | Contrasté cada cifra con `reports/e05_tabla_principal.md` y `e05_resumen.json` | Decisión final elegida por mí entre las opciones planteadas |

**Qué no se aceptó sin verificar:** las cifras se tomaron de mis ejecuciones, no de la IA. La IA señaló que las diferencias de HitRate entre modelos no son concluyentes (error estándar ≈ 0.008) y la conclusión del informe lo refleja.

**Referencias:** solo se citan las del manual (Harper & Konstan, 2015). No se incluyó bibliografía generada sin comprobar.
