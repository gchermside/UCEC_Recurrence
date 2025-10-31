import os
import pandas as pd
import joblib
import config
from sklearn.compose import ColumnTransformer

from preprocessing_utils import (
    load_and_split_data,
    MrnaPreprocessorWrapper,
    ClinicalPreprocessorWrapper,
    MutationPreprocessorWrapper,
)

X_train, y_train, X_val, y_val, X_test, y_test, clinical_cols, mrna_cols, mutation_cols = load_and_split_data()

preprocessor = ColumnTransformer(
    transformers=[
        ("clinical", ClinicalPreprocessorWrapper(), clinical_cols),

        ("mrna", MrnaPreprocessorWrapper(), mrna_cols),

        ("mutation", MutationPreprocessorWrapper(), mutation_cols),
    ]
)

preprocessor.set_output(transform="pandas") # otherwise, output is converted to numpy array

# Create output directory
os.makedirs(config.SPLIT_DATA_DIR, exist_ok=True)

# Save all artifacts in the new folder
joblib.dump(X_train, os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
joblib.dump(y_train, os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
joblib.dump(X_val, os.path.join(config.SPLIT_DATA_DIR, "X_val.joblib"))
joblib.dump(y_val, os.path.join(config.SPLIT_DATA_DIR, "y_val.joblib"))
joblib.dump(X_test, os.path.join(config.SPLIT_DATA_DIR, "X_test.joblib"))
joblib.dump(y_test, os.path.join(config.SPLIT_DATA_DIR, "y_test.joblib"))
joblib.dump(clinical_cols, os.path.join(config.SPLIT_DATA_DIR, "clinical_cols.joblib"))
joblib.dump(mrna_cols, os.path.join(config.SPLIT_DATA_DIR, "mrna_cols.joblib"))
joblib.dump(mutation_cols, os.path.join(config.SPLIT_DATA_DIR, "mutation_cols.joblib"))
joblib.dump(preprocessor, os.path.join(config.SPLIT_DATA_DIR, "preprocessor.joblib"))
