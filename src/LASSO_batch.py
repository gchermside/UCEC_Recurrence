import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline

import joblib
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from preprocessing_utils import *
import config

# Load and split the data ---------------------------------------------------------------

# Define directory where the data was saved

# Load everything
X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
X_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_val.joblib"))
y_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_val.joblib"))
X_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_test.joblib"))
y_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_test.joblib"))
clinical_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "clinical_cols.joblib"))
mrna_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mrna_cols.joblib"))
mutation_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mutation_cols.joblib"))
preprocessor = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "preprocessor.joblib"))

def run_LASSO():
    print("Running Logistic Regression with LASSO")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('var_thresh', VarianceThreshold(threshold=0.001).set_output(transform="pandas")),
        # ("selector", StabilitySelection()),
        # ("selector", BootstrappedSelectKBest()),
        ('select', SelectKBest(score_func=f_classif, k=50)),
        ('clf', LogisticRegression(
            penalty='l1',
            solver='saga',
            class_weight='balanced',      # helps with imbalance
            random_state=config.SEED,
            max_iter=20000,               # more iterations for convergence
            n_jobs=-1,                    # parallelize
            tol=1e-3,
            verbose=0
        ))
    ])

    param_grid = {
        # Regularization strength (C smaller = stronger shrinkage)
        'clf__C': [0.1, 1, 10]
    }

    return pipeline, param_grid

pipeline, param_grid = run_LASSO()

# Set up cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)

# Grid search over pipeline
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',  # could experiment with 'f1' if recurrence class is more important
    n_jobs=1,
    verbose=3
)

# # Fit pipeline on training data
grid_search = grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# Save the trained model

# Create output directory
output_dir = "lasso_grid_search"
os.makedirs(output_dir, exist_ok=True)

# Save all artifacts in the new folder
joblib.dump(best_model, os.path.join(output_dir, "lasso_model.joblib"))
joblib.dump(grid_search, os.path.join(output_dir, "lasso_gridsearch.joblib"))
joblib.dump(X_train, os.path.join(output_dir, "X_train.joblib"))
joblib.dump(y_train, os.path.join(output_dir, "y_train.joblib"))
joblib.dump(X_val, os.path.join(output_dir, "X_val.joblib"))
joblib.dump(y_val, os.path.join(output_dir, "y_val.joblib"))
joblib.dump(X_test, os.path.join(output_dir, "X_test.joblib"))
joblib.dump(y_test, os.path.join(output_dir, "y_test.joblib"))


print(f"All LASSO artifacts saved to '{output_dir}/'")