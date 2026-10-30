import pandas as pd

from inf8239_u03.recommenders import temporal_leave_one_out


def _ratings() -> pd.DataFrame:
    rows = [
        (1, 10, 4.0, 100),
        (1, 11, 3.0, 300),
        (1, 12, 5.0, 200),
        (2, 10, 2.0, 50),
        (2, 13, 4.0, 60),
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_split_reserves_last_interaction_of_each_user():
    _, test = temporal_leave_one_out(_ratings())
    chosen = dict(zip(test["userId"], test["movieId"]))
    assert chosen == {1: 11, 2: 13}


def test_train_has_no_interaction_after_the_test_one():
    train, test = temporal_leave_one_out(_ratings())
    last_train = train.groupby("userId")["timestamp"].max()
    for row in test.itertuples():
        assert last_train[row.userId] <= row.timestamp


def test_train_and_test_are_disjoint_and_complete():
    ratings = _ratings()
    train, test = temporal_leave_one_out(ratings)
    assert len(train) + len(test) == len(ratings)
    keys_train = set(zip(train["userId"], train["movieId"]))
    keys_test = set(zip(test["userId"], test["movieId"]))
    assert not keys_train & keys_test
