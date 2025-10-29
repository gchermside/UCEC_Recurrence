import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from preprocessing_utils import *
import config

# Load everything
X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
X_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_val.joblib"))
y_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_val.joblib"))
X_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_test.joblib"))
y_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_test.joblib"))
preprocessor = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "preprocessor.joblib"))

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)

# -------------------------------------------------------------------------------------
# Utility: Run grid search and save model
# -------------------------------------------------------------------------------------
def run_and_save_model(name, pipeline, param_grid):
    print(f"\n{'='*80}\nRunning model: {name}\n{'='*80}")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=3
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    output_dir = f"{name}_grid_search"
    os.makedirs(output_dir, exist_ok=True)

    # Save all artifacts in the new folder
    joblib.dump(best_model, os.path.join(output_dir, "lasso_model.joblib"))
    joblib.dump(grid_search, os.path.join(output_dir, "lasso_gridsearch.joblib"))

    print(f"All artifacts saved to '{output_dir}/'\n")


# -------------------------------------------------------------------------------------
# 1️⃣ LASSO + SelectKBest
# -------------------------------------------------------------------------------------
def run_select_k_best():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('var_thresh', VarianceThreshold(threshold=0.001).set_output(transform="pandas")),
        ('select', SelectKBest(score_func=f_classif)),
        ('clf', LogisticRegression(
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            random_state=config.SEED,
            max_iter=20000,
            n_jobs=-1,
            tol=1e-3,
        )),
    ])

    param_grid = {
        'select__k': [25, 50, 100, 200],
        'clf__C': [0.01, 0.1, 1, 10]
    }

    run_and_save_model("lasso_selectkbest", pipeline, param_grid)


# -------------------------------------------------------------------------------------
# 2️⃣ LASSO + BootstrappedSelectKBest
# -------------------------------------------------------------------------------------
def run_bootstrapped_select_k_best():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('var_thresh', VarianceThreshold(threshold=0.001).set_output(transform="pandas")),
        ("select", BootstrappedSelectKBest()),
        ('clf', LogisticRegression(
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            random_state=config.SEED,
            max_iter=20000,
            n_jobs=-1,
            tol=1e-3,
        )),
    ])

    param_grid = {
        'select__k': [25, 50, 100],
        'select__n_bootstrap': [20, 50, 100],
        'select__threshold': [0.2, 0.3, 0.4, 0.5, 0.6],
        'clf__C': [0.01, 0.1, 1, 10]
    }

    run_and_save_model("lasso_bootstrapped_selectkbest", pipeline, param_grid)


# -------------------------------------------------------------------------------------
# 3️⃣ LASSO + StabilitySelection
# -------------------------------------------------------------------------------------
def run_stability_selection():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('var_thresh', VarianceThreshold(threshold=0.001).set_output(transform="pandas")),
        ("select", StabilitySelection()),
        ('clf', LogisticRegression(
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            random_state=config.SEED,
            max_iter=20000,
            n_jobs=-1,
            tol=1e-3,
        )),
    ])

    param_grid = {
        'select__stability_threshold': [0.6, 0.7, 0.8, 0.9],
        'select__n_boots': [50, 100],
        'select__fpr_alpha': [0.01, 0.05, 0.1, 0.2],
        'clf__C': [0.01, 0.1, 1, 10]
    }

    run_and_save_model("lasso_stability_selection", pipeline, param_grid)


# -------------------------------------------------------------------------------------
# Run all models sequentially
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    # run_select_k_best()
    run_bootstrapped_select_k_best()
    run_stability_selection()
