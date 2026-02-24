from leakproof_ml.preprocessing.selector import CorrelationSelector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from leakproof_ml import train_test_analysis
from leakproof_ml.validation import ShuffledGroupKFold


def test_correlation_selector(grouped_regression_data):
    X, y, groups = grouped_regression_data
    X["feature2"] = X["feature1"] + 0.01 * (groups + 1)  # Add a highly correlated feature
    
    selector = CorrelationSelector(threshold=0.8)
    # Fit the selector to data
    selector.fit(X, y)
    
    # Transform the data
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == 1  # Only one feature should be retained



def test_custom_pipeline(grouped_regression_data):
    X, y, groups = grouped_regression_data
    
    # Create a simple custom pipeline
    def custom_pipeline(model):
        preprocessor = Pipeline([
            ("scaler", StandardScaler())
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        return pipeline
    
    model = Ridge
    # Splitter
    splitter = ShuffledGroupKFold(n_splits=3, random_state=42)
    # Perform train-test analysis
    results = train_test_analysis(X, y, model, splitter, groups=groups, 
                                  pipeline_factory=custom_pipeline)

    assert results is not None
    