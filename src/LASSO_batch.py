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

X_train, y_train, X_val, y_val, X_test, y_test, clinical_cols, mrna_cols, mutation_cols = load_and_split_data()

preprocessor = ColumnTransformer(
    transformers=[
        ("clinical", ClinicalPreprocessorWrapper(), clinical_cols),

        ("mrna", MrnaPreprocessorWrapper(), mrna_cols),

        ("mutation", MutationPreprocessorWrapper(), mutation_cols),
    ]
)

preprocessor.set_output(transform="pandas") # otherwise, output is converted to numpy array

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
joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
joblib.dump(clinical_cols, os.path.join(output_dir, "clinical_cols.joblib"))
joblib.dump(mrna_cols, os.path.join(output_dir, "mrna_cols.joblib"))
joblib.dump(mutation_cols, os.path.join(output_dir, "mutation_cols.joblib"))
joblib.dump(grid_search, os.path.join(output_dir, "lasso_gridsearch.joblib"))

print(f"All LASSO artifacts saved to '{output_dir}/'")