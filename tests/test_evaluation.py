import math

import pandas as pd

from inf8239_u03.evaluation import ndcg_at_k, pareto_mask, per_user_hits, popularity_share


def _truth():
    return pd.DataFrame({"userId": [1, 2, 3], "movieId": [10, 20, 30]})


def test_per_user_hits_marks_hit_and_miss():
    recs = {1: [10, 11], 2: [21, 22], 3: [31, 30]}
    assert per_user_hits(recs, _truth(), 10) == {1: 1, 2: 0, 3: 1}


def test_per_user_hits_respects_k():
    recs = {3: [31, 32, 30]}
    assert per_user_hits(recs, _truth(), 2) == {3: 0}
    assert per_user_hits(recs, _truth(), 3) == {3: 1}


def test_ndcg_positions():
    recs = {1: [10, 11], 2: [21, 20], 3: [31, 32]}
    expected = (1.0 + 1.0 / math.log2(3) + 0.0) / 3
    assert math.isclose(ndcg_at_k(recs, _truth(), 10), expected)


def test_popularity_share():
    recs = {1: [1, 2, 3, 4], 2: [1, 5, 6, 7]}
    assert popularity_share(recs, {1, 2}) == 3 / 8
    assert popularity_share({}, {1}) == 0.0


def test_pareto_mask_detects_dominated_points():
    frame = pd.DataFrame(
        {
            "hit": [0.05, 0.04, 0.05, 0.03],
            "cov": [0.10, 0.10, 0.08, 0.20],
            "cost": [10.0, 10.0, 20.0, 1.0],
        }
    )
    mask = pareto_mask(frame, maximize=["hit", "cov"], minimize=["cost"])
    # fila 1 la domina la 0 (peor hit, igual cov y costo); fila 2 la domina la 0 (peor cov y mayor costo)
    assert mask.tolist() == [True, False, False, True]


def test_pareto_mask_keeps_identical_points():
    frame = pd.DataFrame({"hit": [0.1, 0.1], "cost": [1.0, 1.0]})
    assert pareto_mask(frame, ["hit"], ["cost"]).tolist() == [True, True]
