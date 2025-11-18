import os
import pandas as pd
import joblib
import config
import numpy as np
from scipy import stats
from itertools import combinations
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

print("saved new split to: ", config.SPLIT_DATA_DIR)

def compare_train_val_test(train_df, val_df, test_df, clinical_cols, save_table=False, table_fn=''):
    """
    Compare clinical variable distributions across training, validation, and test sets.
    Performs pairwise tests and reports significant differences.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Datasets to compare.
    clinical_cols : list of str
        Columns to include in the comparison.
    save_table : bool, optional
        If True, save the output to a CSV.
    table_fn : str, optional
        Filename to save table to if save_table=True.

    Returns
    -------
    pd.DataFrame
        Summary table comparing all sets.
    """
    sets = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
    pair_names = list(combinations(sets.keys(), 2))  # [('Train','Validation'), ('Train','Test'), ('Validation','Test')]
    
    # Separate categorical vs numerical
    categorical_cols = [c for c in clinical_cols if train_df[c].dtype == 'object' or train_df[c].dtype.name == 'category']
    numerical_cols = [c for c in clinical_cols if c not in categorical_cols]

    all_results = []

    # ---------- CATEGORICAL ----------
    for col in categorical_cols:
        combined = pd.concat([
            train_df[[col]].assign(Source='Train'),
            val_df[[col]].assign(Source='Validation'),
            test_df[[col]].assign(Source='Test')
        ])
        combined[col] = combined[col].fillna('NAN')
        
        for (s1, s2) in pair_names:
            subset = combined[combined['Source'].isin([s1, s2])]
            tab = pd.crosstab(subset[col], subset['Source'])
            
            if tab.shape == (2, 2):
                _, pval = stats.fisher_exact(tab)
            else:
                try:
                    _, pval, _, _ = stats.chi2_contingency(tab)
                except Exception:
                    pval = np.nan
            
            sig = "YES" if (pval < 0.05) else "NO"
            all_results.append([col, 'categorical', s1, s2, f"{pval:.3g}", sig])

    # ---------- NUMERICAL ----------
    for col in numerical_cols:
        for (s1, s2) in pair_names:
            vals1 = sets[s1][col].dropna()
            vals2 = sets[s2][col].dropna()
            if len(vals1) == 0 or len(vals2) == 0:
                continue

            try:
                _, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
            except Exception:
                pval = np.nan

            sig = "YES" if (pval < 0.05) else "NO"
            all_results.append([col, 'numeric', s1, s2, f"{pval:.3g}", sig])

    # ---------- COMBINE ----------
    out = pd.DataFrame(all_results, columns=['Variable', 'Type', 'Set 1', 'Set 2', 'p-value', 'Significant (<0.05)'])
    
    if save_table:
        out.to_csv(table_fn or 'train_val_test_comparison.csv', index=False)

    # Print summary of significant differences
    sig_rows = out[out['Significant (<0.05)'] == 'YES']
    if len(sig_rows) > 0:
        print("\nStatistically significant differences detected:")
        for _, row in sig_rows.iterrows():
            print(f" - {row['Variable']} ({row['Type']}): {row['Set 1']} vs {row['Set 2']} (p = {row['p-value']})")
    else:
        print("\nNo statistically significant differences found.")

    return out

summary = compare_train_val_test(
    X_train, X_val, X_test, clinical_cols,
    save_table=True,
    table_fn='train_val_test_comparison_new.csv'
)
