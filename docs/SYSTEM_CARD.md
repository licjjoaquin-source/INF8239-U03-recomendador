# System Card · Recomendador

## Usuarios y propósito

Sistema didáctico que ordena películas para personas con historial de valoraciones de MovieLens. **Uso previsto:** comparar estrategias de recomendación de forma reproducible en un curso. **Fuera de alcance:** cualquier despliegue real, decisiones sobre personas, otros catálogos o dominios. Una lista Top-10 no representa una verdad universal ni garantiza satisfacción.

## Catálogo y candidatos

Catálogo: las películas que aparecen en el conjunto de entrenamiento (9,701). Se excluyen las películas que el usuario ya valoró. El híbrido solo re-ordena los **100 mejores candidatos** de la factorización por usuario.

## Señales utilizadas

- **Colaborativa:** vectores latentes de usuario y película aprendidos de las valoraciones.
- **Contenido:** perfil de géneros del usuario (TF-IDF ponderado por `rating − 2.5`, mínimo 0.1) y similitud coseno con cada película.
- **Popularidad:** conteo de valoraciones y media suavizada hacia la media global.

No se usan datos demográficos ni identificadores externos.

## Modelos y fallback

| Modelo | Descripción |
|---|---|
| Popularidad por conteo y suavizada | Referencias sin personalización |
| Contenido | Géneros; desempate por popularidad porque los empates son masivos |
| Factorización (MF) | SGD, 12 épocas, tasa 0.01, regularización 0.05, semillas 42–44 |
| **Híbrido (seleccionado)** | 20 factores, α = 0.75: `0.75 × colaborativa + 0.25 × contenido` (ambas normalizadas a [0, 1] por usuario) |

**Fallback:** para un usuario sin historial se muestra la popularidad suavizada y se declara como **no personalizada**.

## Métricas offline

Partición: leave-one-out temporal, 587 usuarios evaluados.

| Modelo | HitRate@10 | Cobertura | Entrenamiento | Tamaño de las matrices | Tamaño total |
|---|---|---|---|---|---|
| Popularidad por conteo | 0.0426 | 1.2 % | 0 s | 76 KB | 76 KB |
| Híbrido f=20, α=0.75 | 0.0392 ± 0.0017 | 7.6 % | 10.5 s | 1.6 MB | 3.4 MB |

El tamaño total del híbrido suma los factores de MF (1.6 MB) y la matriz de géneros con su vector IDF (1.8 MB): unas 45 veces el de la popularidad por conteo. El entrenamiento incluye el ajuste de TF-IDF (0.03 s) y la inferencia (0.82 ms por usuario) incluye la construcción de perfiles.

Error estándar del HitRate ≈ 0.008: la diferencia entre ambos **no es concluyente**. RMSE del híbrido: 1.025, frente a 1.116 (media global) y 1.050 (media por película).

## Cold start, cobertura y diversidad

- **Usuario nuevo:** sin historial no hay señal colaborativa ni de contenido; se usa el fallback no personalizado.
- **Historial pequeño o grande:** el HitRate es mayor en el 25 % con menos historial (hasta 34 valoraciones) que en el 25 % con más (desde 161) para casi todos los modelos. Hay pocos aciertos por grupo (de 2 a 9); no se concluyen causas.
- **Cobertura:** el híbrido recomienda el 7.6 % del catálogo; la popularidad, el 1.2 %. Bajar α sube la cobertura pero baja el HitRate.
- **Diversidad:** no se midió diversidad intra-lista de forma directa; la concentración en las 100 películas más populares (57 % frente a 99 %) es una aproximación.

## Riesgos y monitoreo

| Riesgo | Mitigación |
|---|---|
| Popularidad dominante | Medir cobertura y concentración; no elegir un modelo solo por HitRate |
| Burbuja de filtro | Mezclar señales y revisar diversidad de géneros en las listas |
| Sesgo de selección | Declararlo; evaluar solo sobre lo observado |
| Ausencia de demografía | No afirmar resultados por grupo |
| Retroalimentación | Las recomendaciones futuras cambiarían las interacciones de entrenamiento; en un despliegue real habría que registrar la política y medir el efecto |
| Evaluación ruidosa | Reportar error estándar y repetir con varias semillas |

**Monitoreo si se desplegara:** vigilar HitRate, cobertura, concentración en pocas películas, proporción de usuarios fríos y revisión periódica de los datos. Este sistema **no está desplegado**.
