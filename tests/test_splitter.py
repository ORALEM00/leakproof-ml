import numpy as np
from leakproof_ml.validation import ShuffledGroupKFold

def test_no_group_overlap(grouped_regression_data):
    X, y, groups = grouped_regression_data

    splitter = ShuffledGroupKFold(n_splits=3)

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        assert train_groups.isdisjoint(test_groups)


def test_number_of_splits(grouped_regression_data):
    X, y, groups = grouped_regression_data

    splitter = ShuffledGroupKFold(n_splits=3)
    splits = list(splitter.split(X, y, groups))

    assert len(splits) == 3


def test_reproducibility(grouped_regression_data):
    X, y, groups = grouped_regression_data

    splitter1 = ShuffledGroupKFold(n_splits=3, random_state=42)
    splitter2 = ShuffledGroupKFold(n_splits=3, random_state=42)

    splits1 = list(splitter1.split(X, y, groups))
    splits2 = list(splitter2.split(X, y, groups))

    for (train1, test1), (train2, test2) in zip(splits1, splits2):
        assert np.array_equal(train1, train2)
        assert np.array_equal(test1, test2)