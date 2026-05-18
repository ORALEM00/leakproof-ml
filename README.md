
# Leakproof-ml

<p align="center">
  <img width="616" src="./images\visual_abstract.png" />
</p>

---

[![PyPI](https://img.shields.io/pypi/v/leakproof-ml
)](https://pypi.org/project/leakproof-ml/)

[![License](https://img.shields.io/pypi/l/leakproof-ml
)](./LICENSE.txt)

[![Python package](https://github.com/ORALEM00/leakproof-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/ORALEM00/leakproof-ml/actions/workflows/ci.yml)

![Python](https://img.shields.io/pypi/pyversions/leakproof-ml)


**Leakproof ML** is an open-source Python package that enforces leakage-aware machine learning workflows focused on the most common sources of data leakage arising from improper validation strategies and inadequate isolation between training and test data. Designed to systematically prevent data leakage across the complete modelling process, including:

- **Data Splitting**: Requires explicit user-defined splitter.
- **Preprocessing**: Transformations are fit exclusively on training folds. 
- **Feature Engineering**: Chosen within the training fold.
- **Hyperparameter tuning**: Uses nested cross-validation to avoid optimistic bias. 
- **Interpretability**: Feature importance aggregated across test sets only. 

The library is especially designed for small datasets, where improper validation can lead to overoptimistic model performance with misleading interpretability.


## Core Design
Leakproof ML prevents data leakage by implementing two structural constrains: 

- **Explicit splitter**: Unlike most ML workflows where a default random splitter is used, Leakproof ML enforces the explicit definition of a user-defined splitter. This supports a deliberate selection of the validation design and ensures easy comparison between different splitters,  allowing you to choose a splitter strategy more aligned with the data structure to mitigate unintended leakage. Any splitter compatible with scikit-learn's API is supported, including the custom `ShuffledGroupKFold` present in this package.

- **Pipeline use**: All Leakproof ML functions require the use of a scikit-learn `Pipeline` integrated into the modeling process. This guarantees that all preprocessing steps, feature selection, and model fitting procedures are executed exclusively on the current training data. The package provides default pipelines which steps include: standarizartion, feature selection, and model. A custom user-defined can also be implemented. 

## Install

Leakproof ML can be installed from [PyPI](https://pypi.org/project/leakproof-ml/): 

<pre>
pip install leakproof-ml
</pre>


## Quick start
The package provides three main functionalities: training, tuning, and interpretability. Each works with both a standard train-test split and a cross-validation scheme. 

### Setup

```python
# Setting for the example
import pandas as pd
import xgboost
from leakproof_ml.validation import ShuffledGroupKFold

df = pd.read_csv("data.csv")

X = df.drop(columns=["target", "group_id"])
y = df["target"]
# Rows sharing the same group_id are kept together
groups = df["group_id"]

# Define splitter explicitly, any scikit-learn compatible splitter works
splitter = ShuffledGroupKFold(n_splits = 10, random_state = 42)
```

### Training
Fit a model along cross-validation folds (or simple train test) while dimishing data leakage sources. Returns a dictionary with predictions and metrics aggregated across folds.

```python
from leakproof_ml import cv_analysis
from leakproof_ml.plots import plot_predictions

# The class of the model is passed as parameter
# Results are gathered in dictionary format
model_results = cv_analysis(X, y, XGBRegressor, splitter, groups=groups, params = {"max_depth": 4})
# Results dictionary return keys: metrics, y_predict, y_true, features

# For simple train-test split use:
# model_results = train_test_analysis(X, y, XGBRegressor, splitter)

plot_predictions(model_results['y_true'], model_results['y_predict'])
```
<p align="center">
  <img width="616" src="./examples\vanadium_supercapacitor\resulting_plots\XGBRegressor\metrics\groupedCV_predictions.png" />
</p>

### Tuning
Leakproof ML implements a nested CV scheme: an inner loop tunes the parameters, an outer loop evaluates them on held-out data. This avoids the optimistic bias that arises when tuning on the full dataset. For the train-test setting, a CV is applied on the train set to optimize parameters and subsequently evaluated on the held-out test set.

```python
from leakproof_ml.tuning import nested_cv_tuning

# An inner splitter is needed for the hyperparameter search loop
inner_splitter = ShuffledGroupKFold(n_splits = 3, random_state = 42)

# Define the search space using Optuna's trial API
def search_space(trial):
  return {
    "max_depth": trial.suggest_int("max_depth", 2, 5),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
      }

# Returns in addition the set of parameters optimized
tuning_results = nested_cv_tuning(X, y, XGBRegressor, splitter, inner_splitter, search_space, groups=groups) 
# Results dictionary return keys: metrics, y_predict, y_true, params,  features

# For simple train-test
# tuning_results = train_test_tuning(X, y, XGBRegressor, splitter, inner_splitter, search_space)
```

### Interpretability
The package aggregate feature importance calculations across all cross-validation folds, using only the test fold data to avoid interpretability bias. By default, permutation importance (PI) is used; SHAP is also available.

```python
from leakproof_ml.interpretability import cv_interpretability
from leakproof_ml.plots import plot_interpretability_bar 

# Can use parameters calculated in the tuning stage
results = cv_interpretability(X, y, XGBRegressor, splitter, groups=groups, params = tuning_results["params"])
# Results dictionary return keys: metrics, y_predict, y_true, features
# If PI method it addionally include: importance_mean, importance_std
# If shap methods it addionally include: shap_values, X_test

plot_interpretability_bar(results)
```
<p align="center">
  <img width="616" src="./examples\vanadium_supercapacitor\resulting_plots\XGBRegressor\pi\pi_groupedCV.png" />
</p>

## Custom Pipeline
By default, Leakproof ML uses two standard pipelines: (standardization → model) or (standardization → feature selection → model). To ensure flexibility, you can substitute your own custom pipeline by passing a factory function. The function must accept a model as its first argument (can accept more parameters depending on the `Pipeline` steps ) and return a scikit-learn Pipeline whose final step is named `"model"`.

```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from leakproof_ml import cv_analysis

# Custom pipeline 
def polynomial_custom_factory(model, degree = 2):
  numeric_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
  ])
  preprocessor = ColumnTransformer(
    transformers=[
      ('num', numeric_pipe, make_column_selector(dtype_include='float64')),
    ],
    remainder='passthrough'
  )

  # Pipeline steps
  pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=degree)),
    ('model', model)
  ])
  return pipe

results = cv_analysis(X, y, XGBRegressor, splitter, groups=groups, params = {"max_depth": 4}, pipeline_factory = polynomial_custom_factory)
```

## Citation
If used in a research project, please cite paper Leakproof ML in your publication:

<details open>
<summary>BibTeX</summary>

```bibtex
@article{Ortiz2025Leakproof,
  title={Leakproof ML: Data Leakage Prevention with a Robust, Interpretable, and Reproducible Machine Learning Framework},
  author={},
  booktitle={},
  pages={},
  year={}
}
```
</details>

## License

MIT License (see [LICENSE](./LICENSE.txt)).