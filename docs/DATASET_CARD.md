# Dataset Card · MovieLens

## Fuente, fecha, versión y hash

- **Fuente:** GroupLens Research, Universidad de Minnesota. MovieLens Latest Small (`ml-latest-small`).
- **URL de descarga:** https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- **Fecha de descarga:** 29 de octubre de 2026.
- **Hash SHA-256 del archivo ZIP descargado:** `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`
- **Hash SHA-256 de `ratings.csv` local:** `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646`
- **Advertencia de la fuente:** es un conjunto de desarrollo que puede cambiar; no debe tratarse como una versión archivada para resultados científicos compartidos. Por eso se registran fecha y hash.

## Licencia y cita

Uso académico y de investigación bajo las condiciones del README oficial incluido en la descarga (`data/raw/ml-latest-small/README.txt`). Resumen de lo que implica para este trabajo (confirmar siempre con el README local): citar la obra, no insinuar respaldo de la Universidad de Minnesota o de GroupLens, **no redistribuir los datos** y no usarlos con fines comerciales sin permiso. Por eso el repositorio no contiene los datos: solo el script de descarga.

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), 1–19. https://doi.org/10.1145/2827872

## Archivos y diccionario

Archivos usados:

| Archivo | Columnas | Descripción |
|---|---|---|
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | Valoración de 0.5 a 5 estrellas (pasos de 0.5) y marca temporal Unix en segundos |
| `movies.csv` | `movieId`, `title`, `genres` | Título con año y géneros separados por `\|` |

El ZIP incluye además `tags.csv` y `links.csv`, que este trabajo no utiliza.

## Procedimiento de descarga

`uv run python scripts/download_data.py` descarga el ZIP, calcula el hash, valida las rutas del archivo comprimido y lo extrae en `data/raw/`. `uv run python scripts/audit_data.py` verifica columnas, claves foráneas, rango de ratings y marcas temporales.

## Población observada y exclusiones

| Dato | Valor |
|---|---|
| Valoraciones | 100,836 |
| Usuarios | 610 |
| Películas en el catálogo (`movies.csv`) | 9,742 |
| Películas con al menos una valoración | 9,724 |
| Densidad de la matriz usuario–película | 1.70 % |
| Rating medio / mediana / desviación | 3.50 / 3.5 / 1.04 |
| Rango temporal | 29/03/1996 a 24/09/2018 |

- Los usuarios incluidos habían valorado al menos 20 películas: no existen usuarios con historial casi nulo en el conjunto.
- **Exclusiones del experimento:** la última valoración de cada usuario se reserva para evaluación (610 valoraciones; 100,226 quedan en entrenamiento, con 9,701 películas). Se evalúan 587 usuarios; los otros 23 se excluyen porque su película retenida no aparece en entrenamiento.

## Calidad, sesgos y usos prohibidos

- **Autoselección:** solo existen valoraciones de películas que cada persona decidió ver; la ausencia de valoración no significa rechazo (densidad de solo 1.7 %).
- **Sesgo de popularidad:** pocas películas concentran muchas valoraciones, y un recomendador por popularidad se ve favorecido.
- **Sin demografía:** los identificadores son anónimos. No se pueden afirmar resultados por grupos poblacionales ni inferir atributos personales.
- **Cobertura temporal y cultural:** datos hasta 2018 y de usuarios de una plataforma concreta; no representan a toda la población ni al catálogo actual.
- **Usos prohibidos:** redistribuir los datos, usarlos comercialmente sin permiso, intentar reidentificar usuarios y presentar los resultados como válidos para otro catálogo, población o dominio.
