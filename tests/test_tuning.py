from sklearn.linear_model import Ridge
from leakproof_ml.validation import ShuffledGroupKFold
from leakproof_ml.tuning import train_test_tunning, nested_cv_tunning


def space_search(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
    }

def test_train_test_tunning(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    outer_splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    inner_splitter = ShuffledGroupKFold(n_splits=2, random_state=42)
    # Perform train-test tuning
    results = train_test_tunning(X, y, model, outer_splitter, inner_splitter, 
                                 groups=groups, space_search=space_search)

    assert results is not None

def test_nested_cv_tunning(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    outer_splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    inner_splitter = ShuffledGroupKFold(n_splits=2, random_state=42)
    # Perform nested CV tuning
    results = nested_cv_tunning(X, y, model, outer_splitter, inner_splitter, 
                                groups=groups, space_search=space_search)

    assert results is not None