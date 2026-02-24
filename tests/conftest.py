# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def grouped_regression_data():
    rng = np.random.RandomState(42)

    n_groups = 6
    samples_per_group = 4

    X, y, groups = [], [], []

    for g in range(n_groups):
        group_signal = rng.normal()

        for _ in range(samples_per_group):
            X.append([group_signal + rng.normal(scale=0.1)])
            y.append(group_signal)
            groups.append(g)

    return (
        pd.DataFrame(np.array(X), columns=["feature1"]),
        pd.Series(np.array(y)),
        np.array(groups),
        )