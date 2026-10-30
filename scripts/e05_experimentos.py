"""Ejercicio 05 · Comparación reproducible de recomendadores.

Compara, con la misma partición temporal (la última valoración de cada usuario
queda para evaluación), cuatro familias de modelos:

* popularidad (por conteo y suavizada),
* contenido (perfil de géneros del usuario con TF-IDF y coseno),
* colaborativo (factorización matricial),
* híbrido (re-ordena los 100 mejores candidatos colaborativos mezclando la
  señal colaborativa y la de contenido con el peso alpha).

Repite el entrenamiento con varias semillas y varios números de factores, mide
utilidad (HitRate@10, NDCG@10, cobertura), costo (tiempo y tamaño), segmenta por
tamaño de historial, muestra tres perfiles de usuario (amplio, pequeño y nuevo)
y calcula la frontera de Pareto.

Uso:
    uv run python scripts/e05_experimentos.py
    uv run python scripts/e05_experimentos.py --seeds 42 43 44 --factors 10 20 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error

from inf8239_u03.config import MOVIELENS_DIR, RANDOM_STATE, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.evaluation import ndcg_at_k, pareto_mask, per_user_hits, popularity_share
from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k
from inf8239_u03.recommenders import MatrixFactorization, temporal_leave_one_out, weighted_popularity

K = 10          # longitud de la lista recomendada
POOL = 100      # candidatos colaborativos que el híbrido re-ordena
DETERMINISTIC = -1  # marca de semilla para modelos sin aleatoriedad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2])
    parser.add_argument("--factors", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--demo-factors", type=int, default=20, help="factores del modelo usado en los perfiles de usuario")
    parser.add_argument("--demo-alpha", type=float, default=0.75, help="alpha del híbrido usado en los perfiles de usuario")
    return parser.parse_args()


def minmax(values: np.ndarray) -> np.ndarray:
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-9)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # evita UnicodeEncodeError con α al redirigir la salida en Windows
    args = parse_args()
    started = perf_counter()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ datos y corte temporal
    ratings, movies = load_movielens(MOVIELENS_DIR)
    ratings_sha = hashlib.sha256((MOVIELENS_DIR / "ratings.csv").read_bytes()).hexdigest()
    train, test = temporal_leave_one_out(ratings)
    users = sorted(train["userId"].unique())
    items = sorted(train["movieId"].unique())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: i for i, m in enumerate(items)}
    items_arr = np.asarray(items)
    n_users, n_items = len(users), len(items)

    evaluable = test[test["userId"].isin(user_index) & test["movieId"].isin(item_index)].reset_index(drop=True)
    eval_users = evaluable["userId"].to_numpy()
    eval_rows = np.array([user_index[u] for u in eval_users])
    n_eval = len(evaluable)
    truth_items = np.array([item_index[m] for m in evaluable["movieId"]])

    u_idx = train["userId"].map(user_index).to_numpy()
    i_idx = train["movieId"].map(item_index).to_numpy()
    seen = np.zeros((n_users, n_items), dtype=bool)
    seen[u_idx, i_idx] = True
    item_count = np.bincount(i_idx, minlength=n_items)
    history_size = np.bincount(u_idx, minlength=n_users)[eval_rows]
    top_popular_100 = {int(items_arr[j]) for j in np.argsort(-item_count, kind="stable")[:100]}

    # segmentos por tamaño de historial (cuartiles entre usuarios evaluables)
    q1, q3 = np.quantile(history_size, [0.25, 0.75])
    segment_of = {
        int(u): ("historial pequeño (<=Q1)" if h <= q1 else "historial grande (>=Q3)" if h >= q3 else "intermedio")
        for u, h in zip(eval_users, history_size)
    }

    # ------------------------------------------------------------------ referencias de rating
    y_true = evaluable["rating"].to_numpy()
    global_mean = float(train["rating"].mean())
    item_mean = train.groupby("movieId")["rating"].mean()
    rmse_global = float(np.sqrt(np.mean((y_true - global_mean) ** 2)))
    rmse_item_mean = float(np.sqrt(np.mean((y_true - evaluable["movieId"].map(item_mean).fillna(global_mean).to_numpy()) ** 2)))

    # ------------------------------------------------------------------ contenido: perfil de géneros por usuario
    movie_table = movies.reset_index(drop=True).copy()
    genres_text = movie_table["genres"].str.replace("|", " ", regex=False)
    genre_all = TfidfVectorizer().fit_transform(genres_text).toarray()
    movie_row = {int(m): i for i, m in enumerate(movie_table["movieId"])}
    genre = genre_all[[movie_row[int(m)] for m in items]]
    weights = np.clip(train["rating"].to_numpy() - 2.5, 0.1, None)
    weight_matrix = np.zeros((n_users, n_items))
    weight_matrix[u_idx, i_idx] = weights
    profile = (weight_matrix[eval_rows] @ genre) / weight_matrix[eval_rows].sum(axis=1, keepdims=True)
    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    content_eval = (profile @ genre.T) / norms  # filas de `genre` ya tienen norma 1 (TF-IDF con norma L2)
    del weight_matrix

    # ------------------------------------------------------------------ utilidades de evaluación
    rows: list[dict] = []
    segment_rows: list[dict] = []

    def evaluate(model: str, factors, alpha, seed: int, recs: dict[int, list[int]], **extra) -> None:
        hit = hit_rate_at_k(recs, evaluable, K)
        rows.append(
            {
                "modelo": model, "factores": factors, "alpha": alpha, "semilla": seed,
                "hit_rate_at_10": hit, "hit_se": float(np.sqrt(hit * (1 - hit) / n_eval)),
                "ndcg_at_10": ndcg_at_k(recs, evaluable, K),
                "cobertura": catalog_coverage(recs, n_items),
                "participacion_top100_popular": popularity_share(recs, top_popular_100),
                "n_evaluados": n_eval, **extra,
            }
        )
        hits = per_user_hits(recs, evaluable, K)
        frame = pd.DataFrame({"userId": list(hits), "hit": list(hits.values())})
        frame["segmento"] = frame["userId"].map(segment_of)
        for segment, group in frame.groupby("segmento"):
            segment_rows.append(
                {"modelo": model, "factores": factors, "alpha": alpha, "semilla": seed,
                 "segmento": segment, "hit_rate_at_10": float(group["hit"].mean()), "n_usuarios": len(group)}
            )

    def lists_from_order(order_idx: np.ndarray) -> dict[int, list[int]]:
        recs = {}
        for user_id, row in zip(eval_users, eval_rows):
            recs[int(user_id)] = [int(items_arr[j]) for j in order_idx[~seen[row, order_idx]][:K]]
        return recs

    def lists_from_matrix(index_matrix: np.ndarray) -> dict[int, list[int]]:
        return {int(u): [int(m) for m in items_arr[index_matrix[r, :K]]] for r, u in enumerate(eval_users)}

    # ------------------------------------------------------------------ modelos deterministas
    t0 = perf_counter()
    stats = train.groupby("movieId")["rating"].agg(["mean", "count"]).sort_values(["count", "mean"], ascending=False)
    recs = lists_from_order(np.array([item_index[m] for m in stats.index]))
    elapsed = perf_counter() - t0
    evaluate("popularidad_conteo", 0, np.nan, DETERMINISTIC, recs, rmse=np.nan, train_seconds=0.0,
             infer_ms_por_usuario=1000 * elapsed / n_eval, tamano_kb=item_count.nbytes / 1024)

    t0 = perf_counter()
    smoothed = weighted_popularity(train, movies)
    smoothed_order = np.array([item_index[m] for m in smoothed.index if m in item_index])
    recs = lists_from_order(smoothed_order)
    elapsed = perf_counter() - t0
    evaluate("popularidad_suavizada", 0, np.nan, DETERMINISTIC, recs, rmse=np.nan, train_seconds=0.0,
             infer_ms_por_usuario=1000 * elapsed / n_eval, tamano_kb=smoothed_order.nbytes / 1024)

    t0 = perf_counter()
    content_index = np.zeros((n_eval, K), dtype=int)
    for r, row in enumerate(eval_rows):
        masked = np.where(seen[row], -np.inf, content_eval[r])
        # desempate por popularidad: los empates de géneros son masivos en este catálogo
        content_index[r] = np.lexsort((-item_count, -np.round(masked, 9)))[:K]
    elapsed = perf_counter() - t0
    evaluate("contenido", 0, np.nan, DETERMINISTIC, lists_from_matrix(content_index), rmse=np.nan, train_seconds=0.0,
             infer_ms_por_usuario=1000 * elapsed / n_eval, tamano_kb=genre.nbytes / 1024)

    # ------------------------------------------------------------------ modelos con semilla: MF y híbrido
    demo_cache: dict | None = None
    demo_model: MatrixFactorization | None = None
    for factors in args.factors:
        for seed in args.seeds:
            t0 = perf_counter()
            model = MatrixFactorization(factors=factors, seed=seed).fit(train, epochs=args.epochs)
            train_seconds = perf_counter() - t0
            assert model.items == items and model.users == users, "índices del modelo distintos a los del script"
            size_kb = (model.user_factors.nbytes + model.item_factors.nbytes) / 1024
            predictions = model.global_mean + np.einsum(
                "ij,ij->i", model.user_factors[eval_rows], model.item_factors[truth_items]
            )
            rmse = float(root_mean_squared_error(y_true, predictions))

            t0 = perf_counter()
            scores = model.global_mean + model.user_factors[eval_rows] @ model.item_factors.T
            scores = np.where(seen[eval_rows], -np.inf, scores)
            top_idx = np.argsort(-scores, axis=1, kind="stable")[:, :POOL]
            cf_scores = np.take_along_axis(scores, top_idx, axis=1)
            t_cf = perf_counter() - t0
            evaluate("colaborativo_mf", factors, np.nan, seed, lists_from_matrix(top_idx), rmse=rmse,
                     train_seconds=train_seconds, infer_ms_por_usuario=1000 * t_cf / n_eval, tamano_kb=size_kb)

            t0 = perf_counter()
            norm_cf = minmax(cf_scores)
            norm_content = minmax(np.take_along_axis(content_eval, top_idx, axis=1))
            t_prep = perf_counter() - t0
            for alpha in args.alphas:
                t0 = perf_counter()
                hybrid = alpha * norm_cf + (1 - alpha) * norm_content
                order = np.argsort(-hybrid, axis=1, kind="stable")[:, :K]  # empates: conserva el orden colaborativo
                chosen = np.take_along_axis(top_idx, order, axis=1)
                t_re = perf_counter() - t0
                evaluate("hibrido", factors, alpha, seed, lists_from_matrix(chosen), rmse=rmse,
                         train_seconds=train_seconds,
                         infer_ms_por_usuario=1000 * (t_cf + t_prep + t_re) / n_eval, tamano_kb=size_kb)
                if factors == args.demo_factors and seed == args.seeds[0] and alpha == args.demo_alpha:
                    demo_cache = {"chosen": chosen}
                    demo_model = model
            print(f"  listo: factores={factors} semilla={seed} ({train_seconds:.1f} s de entrenamiento)", flush=True)

    # ------------------------------------------------------------------ tablas
    detail = pd.DataFrame(rows)
    detail["costo_segundos"] = detail["train_seconds"] + detail["infer_ms_por_usuario"] * n_eval / 1000
    detail.to_csv(out / "e05_resultados_por_semilla.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(out / "e05_segmentos.csv", index=False)

    keys = ["modelo", "factores", "alpha"]
    metric_cols = ["hit_rate_at_10", "ndcg_at_10", "cobertura", "participacion_top100_popular", "rmse",
                   "train_seconds", "infer_ms_por_usuario", "costo_segundos", "tamano_kb"]
    grouped = detail.groupby(keys, dropna=False)[metric_cols]
    summary = grouped.mean().add_suffix("_media").join(grouped.std().fillna(0.0).add_suffix("_std")).reset_index()
    summary["n_semillas"] = detail.groupby(keys, dropna=False)["semilla"].nunique().to_numpy()
    summary["hit_se"] = np.sqrt(summary["hit_rate_at_10_media"] * (1 - summary["hit_rate_at_10_media"]) / n_eval)
    summary["pareto"] = pareto_mask(
        summary, maximize=["hit_rate_at_10_media", "cobertura_media"], minimize=["costo_segundos_media"]
    )
    # Pareto bidimensional: utilidad (HitRate) frente a costo, sin la cobertura como tercer objetivo
    summary["pareto_2d"] = pareto_mask(summary, maximize=["hit_rate_at_10_media"], minimize=["costo_segundos_media"])
    summary = summary.sort_values("hit_rate_at_10_media", ascending=False).reset_index(drop=True)
    summary.to_csv(out / "e05_tabla_principal.csv", index=False)
    summary[summary["pareto"]].to_csv(out / "e05_pareto.csv", index=False)

    def label(r) -> str:
        if r["modelo"] == "hibrido":
            return f"híbrido f={int(r['factores'])} α={r['alpha']:g}"
        if r["modelo"] == "colaborativo_mf":
            return f"MF f={int(r['factores'])}"
        return r["modelo"].replace("_", " ")

    summary["configuracion"] = summary.apply(label, axis=1)
    lines = ["| Configuración | HitRate@10 | NDCG@10 | Cobertura | RMSE | Entrenamiento (s) | Inferencia (ms/usuario) | Tamaño (KB) | Pareto 3 obj. | Pareto 2 obj. |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        rmse_txt = "n/a" if pd.isna(r["rmse_media"]) else f"{r['rmse_media']:.3f}"
        lines.append(
            f"| {r['configuracion']} | {r['hit_rate_at_10_media']:.4f} ± {r['hit_rate_at_10_std']:.4f} "
            f"| {r['ndcg_at_10_media']:.4f} | {r['cobertura_media']:.3f} | {rmse_txt} "
            f"| {r['train_seconds_media']:.1f} | {r['infer_ms_por_usuario_media']:.2f} "
            f"| {r['tamano_kb_media']:.0f} | {'sí' if r['pareto'] else 'no'} | {'sí' if r['pareto_2d'] else 'no'} |"
        )
    (out / "e05_tabla_principal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ figura de Pareto
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, r in summary.iterrows():
        on = bool(r["pareto"])
        ax.scatter(r["costo_segundos_media"], r["hit_rate_at_10_media"], s=40 + 600 * r["cobertura_media"],
                   color="#1f4e79" if on else "#b0b7c0", alpha=0.85, edgecolor="black" if on else "none", zorder=3)
        if bool(r["pareto_2d"]):
            ax.scatter(r["costo_segundos_media"], r["hit_rate_at_10_media"], s=220, marker="D", facecolor="none",
                       edgecolor="#c0392b", linewidth=1.8, zorder=4)
        if on:
            ax.annotate(r["configuracion"], (r["costo_segundos_media"], r["hit_rate_at_10_media"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Costo de cómputo (segundos: entrenamiento + inferencia a todos los usuarios; escala simlog)")
    ax.set_ylabel("HitRate@10 (media de semillas)")
    ax.set_title("Pareto: azul = no dominadas con 3 objetivos; rombo rojo = frontera HitRate vs costo\n(tamaño del punto = cobertura)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "e05_pareto.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ perfiles de usuario (amplio, pequeño, nuevo)
    profile_rows: list[dict] = []
    title_of = movies.set_index("movieId")["title"]
    genres_of = movies.set_index("movieId")["genres"]
    if demo_cache is not None and demo_model is not None:
        heavy = int(np.argmax(history_size))
        light = int(np.argmin(history_size))
        for name, r in (("historial amplio", heavy), ("historial pequeño", light)):
            user_id = int(eval_users[r])
            seen_items = set(train.loc[train["userId"].eq(user_id), "movieId"])
            top_genres = (
                train.loc[train["userId"].eq(user_id), "movieId"].map(genres_of).str.split("|").explode().value_counts().head(3)
            )
            chosen_ids = [int(m) for m in items_arr[demo_cache["chosen"][r]]]
            assert not seen_items.intersection(chosen_ids), "el híbrido recomendó un ítem ya visto"
            for rank, movie_id in enumerate(chosen_ids, start=1):
                profile_rows.append(
                    {"perfil": name, "userId": user_id, "n_historial": int(history_size[r]),
                     "generos_frecuentes": ", ".join(top_genres.index), "metodo": f"híbrido α={args.demo_alpha:g}, f={args.demo_factors}",
                     "posicion": rank, "movieId": movie_id, "titulo": title_of[movie_id], "generos": genres_of[movie_id],
                     "nota": "personalizada; excluye ítems vistos"}
                )
    for rank, movie_id in enumerate(smoothed.index[:K], start=1):
        profile_rows.append(
            {"perfil": "usuario nuevo", "userId": int(ratings["userId"].max()) + 1, "n_historial": 0,
             "generos_frecuentes": "", "metodo": "fallback: popularidad suavizada", "posicion": rank,
             "movieId": int(movie_id), "titulo": title_of[movie_id], "generos": genres_of[movie_id],
             "nota": "NO personalizada: sin historial no hay señal colaborativa ni de contenido"}
        )
    pd.DataFrame(profile_rows).to_csv(out / "e05_perfiles.csv", index=False)

    # ------------------------------------------------------------------ resumen y reproducibilidad
    meta = {
        "dataset": "MovieLens ml-latest-small (GroupLens); los datos no se redistribuyen",
        "ratings_sha256_local": ratings_sha,
        "particion": "leave-one-out temporal: la última valoración de cada usuario se reserva para evaluación",
        "usuarios": n_users, "items_en_entrenamiento": n_items, "usuarios_evaluados": n_eval,
        "valoraciones_entrenamiento": len(train), "k": K, "candidatos_hibrido": POOL,
        "semillas": args.seeds, "factores": args.factors, "alphas": args.alphas, "epocas": args.epochs,
        "rmse_referencia_media_global": rmse_global, "rmse_referencia_media_por_pelicula": rmse_item_mean,
        "desempate_contenido": "por popularidad (conteo en entrenamiento)",
        "umbrales_historial": {"q1": float(q1), "q3": float(q3)},
        "entorno": {
            "python": sys.version.split()[0], "sistema": platform.platform(), "procesador": platform.processor(),
            "nucleos": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "segundos_totales": perf_counter() - started,
    }
    (out / "e05_resumen.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nRMSE de referencia: media global={rmse_global:.3f}, media por película={rmse_item_mean:.3f}")
    print(f"Usuarios evaluados: {n_eval}. Archivos en: {out}")


if __name__ == "__main__":
    main()
