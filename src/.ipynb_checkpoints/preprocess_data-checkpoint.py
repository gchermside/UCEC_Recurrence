import os
import joblib
import config
from preprocessing_utils import *

X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
X_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_val.joblib"))
y_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_val.joblib"))
X_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_test.joblib"))
y_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_test.joblib"))
clinical_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "clinical_cols.joblib"))
mrna_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mrna_cols.joblib"))
mutation_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mutation_cols.joblib"))

# Split modalities
clinical_train = X_train[clinical_cols]
mrna_train = X_train[mrna_cols]
mutation_train = X_train[mutation_cols]

clinical_val = X_val[clinical_cols]
mrna_val = X_val[mrna_cols]
mutation_val = X_val[mutation_cols]

clinical_test = X_test[clinical_cols]
mrna_test = X_test[mrna_cols]
mutation_test = X_test[mutation_cols]

print(mrna_train.shape)
print(mutation_train.shape)
print(clinical_train.shape)

# === Initialize and fit each preprocessor on training data ===
clinical_prep = ClinicalPreprocessorWrapper(
    cols_to_remove=config.CLINICAL_COLS_TO_REMOVE,
    categorical_cols=config.CATEGORICAL_COLS,
    max_null_frac=config.CLINICAL_MAX_NULL_FRAC,
    uniform_thresh=config.CLINICAL_UNIFORM_THRESH,
)
mrna_prep = MrnaPreprocessorWrapper(
    max_null_frac=config.MAX_NULL_FRAC,
    uniform_thresh=config.UNIFORM_THRESHOLD,
    corr_thresh=config.CORRELATION_THRESHOLD,
    var_thresh=config.VARIANCE_THRESHOLD,
    re_run_pruning=config.RE_RUN_PRUNING,
    literature_genes=config.LITERATURE_GENES,
    correlated_genes_path=config.CORRELATED_GENES_PATH,
    use_stability_selection=config.USE_STABILITY_SELECTION, # NOTE: For this run, we are going to stability selection
    n_boots=config.N_BOOTS_FPR, # NOTE: might want to experiment with these values, they are set pretty strict right now and I'm not sure that is good for pytorch
    fpr_alpha=0.2, #FIXME: put back to config
    stability_threshold=0.75, # FIXME: put this back to config
    random_state=config.SEED,
)
mutation_prep = MutationPreprocessorWrapper(
    max_null_frac=config.MUTATION_MAX_NULL_FRAC,
    uniform_thresh=config.MUTATION_UNIFORM_THRESH,
)

# Fit on train
clinical_prep.fit(clinical_train)
mrna_prep.fit(mrna_train, y_train)
mutation_prep.fit(mutation_train)

# Transform train, val, test
clinical_train = clinical_prep.transform(clinical_train)
clinical_val = clinical_prep.transform(clinical_val)
clinical_test = clinical_prep.transform(clinical_test)

mrna_train = mrna_prep.transform(mrna_train)
mrna_val = mrna_prep.transform(mrna_val)
mrna_test = mrna_prep.transform(mrna_test)

mutation_train = mutation_prep.transform(mutation_train)
mutation_val = mutation_prep.transform(mutation_val)
mutation_test = mutation_prep.transform(mutation_test)


stability_selection_mrna = StabilitySelection(n_boots=config.N_BOOTS_FPR,
                                         fpr_alpha=0.05,
                                         stability_threshold=0.80,
                                         random_state=config.SEED)

stability_selection_mrna.fit(mrna_train, y_train)
mrna_train = stability_selection_mrna.transform(mrna_train)
mrna_val = stability_selection_mrna.transform(mrna_val)
mrna_test = stability_selection_mrna.transform(mrna_test)

stability_selection_mutation = StabilitySelection(n_boots=config.N_BOOTS_FPR,
                                         fpr_alpha=0.05,
                                         stability_threshold=0.70,
                                         random_state=config.SEED)

stability_selection_mutation.fit(mutation_train, y_train)
mutation_train = stability_selection_mutation.transform(mutation_train)
mutation_val = stability_selection_mutation.transform(mutation_val)
mutation_test = stability_selection_mutation.transform(mutation_test)

print(mrna_train.shape)
print(mutation_train.shape)
print(clinical_train.shape)


# === Making directory ===
base_dir = "../preprocessed_data/no_feature_selection"
for split in ["train", "val", "test"]:
    os.makedirs(f"{base_dir}/{split}", exist_ok=True)

# # === Save all 12 datasets ===
joblib.dump(clinical_train, f"{base_dir}/train/clinical.pkl")
joblib.dump(mrna_train, f"{base_dir}/train/mrna.pkl")
joblib.dump(mutation_train, f"{base_dir}/train/mutation.pkl")
joblib.dump(y_train, f"{base_dir}/train/labels.pkl")

joblib.dump(clinical_val, f"{base_dir}/val/clinical.pkl")
joblib.dump(mrna_val, f"{base_dir}/val/mrna.pkl")
joblib.dump(mutation_val, f"{base_dir}/val/mutation.pkl")
joblib.dump(y_val, f"{base_dir}/val/labels.pkl")

joblib.dump(clinical_test, f"{base_dir}/test/clinical.pkl")
joblib.dump(mrna_test, f"{base_dir}/test/mrna.pkl")
joblib.dump(mutation_test, f"{base_dir}/test/mutation.pkl")
joblib.dump(y_test, f"{base_dir}/test/labels.pkl")
