from sklearn.linear_model import Ridge
from leakproof_ml.validation import ShuffledGroupKFold
from leakproof_ml.interpretability import train_test_interpretability, cv_interpretability

def test_train_test_interpretability(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform train-test interpretability analysis
    results = train_test_interpretability(X, y, model, splitter, groups=groups)

    assert results is not None


def test_train_test_shap_interpretability(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform train-test SHAP interpretability analysis
    results = train_test_interpretability(X, y, model, splitter, groups=groups, method="shap")

    assert results is not None


def test_cv_interpretability(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform cross-validation interpretability analysis
    results = cv_interpretability(X, y, model, splitter, groups=groups)

    assert results is not None


def test_cv_shap_interpretability(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform cross-validation interpretability analysis
    results = cv_interpretability(X, y, model, splitter, groups=groups, method="shap")

    assert results is not None