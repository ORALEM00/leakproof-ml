from sklearn.linear_model import Ridge
from leakproof_ml.validation import ShuffledGroupKFold
from leakproof_ml import train_test_analysis, cv_analysis

def test_train_test_analysis(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform train-test analysis
    results = train_test_analysis(X, y, model, splitter, groups=groups)

    assert results is not None


def test_cv_analysis(grouped_regression_data):
    X, y, groups = grouped_regression_data
    # Create a simple model
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform cross-validation analysis
    results = cv_analysis(X, y, model, splitter, groups=groups)

    assert results is not None


