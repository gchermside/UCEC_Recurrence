import numpy as np
import pandas as pd
import joblib
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


from preprocessing_utils import *
import config

# Load everything
X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
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
    joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(grid_search, os.path.join(output_dir, "gridsearch.joblib"))

    print(f"All artifacts saved to '{output_dir}/'\n")


# -------------------------------------------------------------------------------------
# LASSO + SelectKBest
# -------------------------------------------------------------------------------------
def run_lasso_selectkbest():
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
# LASSO + BootstrappedSelectKBest
# -------------------------------------------------------------------------------------
def run_lasso_bootstrapped_selectkbest():
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
# LASSO + StabilitySelection
# -------------------------------------------------------------------------------------
def run_lasso_stability_selection():
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
# LASSO no feature selection
# -------------------------------------------------------------------------------------
def run_lasso_no_feature_selection():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('var_thresh', VarianceThreshold(threshold=0.001).set_output(transform="pandas")),
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
        'clf__C': [0.01, 0.1, 1, 10]
    }

    run_and_save_model("lasso_no_feature_selection", pipeline, param_grid)

# -------------------------------------------------------------------------------------
# XGBoost (no feature selction)
# -------------------------------------------------------------------------------------
def run_xgboost():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_jobs=-1,
            random_state=config.SEED,
            tree_method='hist',        # Efficient, GPU-friendly if available
            use_label_encoder=False
        )),
    ])

    # Larger but structured grid for balanced exploration
    param_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__subsample': [0.7, 0.85, 1.0],
        'clf__colsample_bytree': [0.7, 0.85, 1.0],
        'clf__gamma': [0, 0.1, 0.3],
    }

    run_and_save_model("xgboost", pipeline, param_grid)

# -------------------------------------------------------------------------------------
# XGBoost (no feature selction)
# -------------------------------------------------------------------------------------
def run_xgboost_selectkbest():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('select', SelectKBest(score_func=f_classif)),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_jobs=-1,
            random_state=config.SEED,
            tree_method='hist',        # Efficient, GPU-friendly if available
            use_label_encoder=False
        )),
    ])
    print(type(pipeline.named_steps['clf']))

    # Larger but structured grid for balanced exploration
    param_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__learning_rate': [0.05, 0.1],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
        'select__k': [25, 50, 100, 200]
    }


    run_and_save_model("xgboost_selectkbest", pipeline, param_grid)

# -------------------------------------------------------------------------------------
# Random Forest (SelectKBest)
# -------------------------------------------------------------------------------------

def run_randomforest_selectkbest():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('select', SelectKBest(score_func=f_classif)),
        ('clf', RandomForestClassifier(
            n_jobs=-1,
            random_state=config.SEED
        )),
    ])

    # Structured but smaller grid for reasonable runtime
    param_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2'],
        'select__k': [25, 50, 100, 200]
    }

    run_and_save_model("randomforest_selectkbest", pipeline, param_grid)

# -------------------------------------------------------------------------------------
# Random Forest (Stability Selection)
# -------------------------------------------------------------------------------------

def run_randomforest_stability_selection():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('select', StabilitySelection()),
        ('clf', RandomForestClassifier(
            n_jobs=-1,
            random_state=config.SEED
        )),
    ])

    # Structured but smaller grid for reasonable runtime
    param_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2'],
        'select__stability_threshold': [0.7, 0.8, 0.9],
        'select__n_boots': [50],
        'select__fpr_alpha': [0.01, 0.05],
    }

    run_and_save_model("randomforest_selectkbest", pipeline, param_grid)

# -------------------------------------------------------------------------------------
# SVC (SelectKBest)
# -------------------------------------------------------------------------------------

from sklearn.svm import SVC

def run_svc_selectkbest():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('select', SelectKBest(score_func=f_classif)),
        ('clf', SVC(
            probability=True,      # Needed for ROC/AUC
            random_state=config.SEED
        )),
    ])

    # SVC grid:
    param_grid = {
        'select__k': [25, 50, 100, 200],
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ['scale', 'auto'],  # Relevant for rbf
    }

    run_and_save_model("svm_selectkbest", pipeline, param_grid)


# -------------------------------------------------------------------------------------
# Run all models
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else None

    if model_name == "lasso_no_feature_selection":
        run_lasso_no_feature_selection()
    elif model_name == "lasso_selectkbest":
        run_lasso_selectkbest()
    elif model_name == "lasso_bootstrapped_selectkbest":
        run_lasso_bootstrapped_selectkbest()
    elif model_name == "lasso_stability_selection":
        run_lasso_stability_selection()
    elif model_name == "xgboost_selectkbest":
        run_xgboost_selectkbest()
    elif model_name == "xgboost":
        run_xgboost()
    elif model_name == "randomforest_selectkbest":
        run_randomforest_selectkbest()
    elif model_name == "randomforest_stability_selection":
        run_randomforest_stability_selection()
    elif model_name == "svc_selectkbest":
        run_svc_selectkbest()
    else:
        print("Please provide a valid model name.")
