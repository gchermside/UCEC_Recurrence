# Imports
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Checkpoint
import joblib
import random
import pickle
import numpy as np
from statistics import mean, stdev
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, f1_score, average_precision_score
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import config
from preprocessing_utils import *
from model_utils import *

# Set seeds for reproducibility
set_random_seed(config.SEED, deterministic=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load splits and column lists
X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
clinical_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "clinical_cols.joblib"))
mrna_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mrna_cols.joblib"))
mutation_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mutation_cols.joblib"))

# Split modalities
clinical_train = X_train[clinical_cols]
mrna_train = X_train[mrna_cols]
mutation_train = X_train[mutation_cols]

def to_loader(clinical, mrna, mutation, y, shuffle=False):
    """
    Convert pandas DataFrames / numpy arrays to a PyTorch DataLoader.
    Includes a numeric check and prints warnings if non-numeric columns
    are present.
    """
    def check_numeric(df, name):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        if non_numeric_cols:
            print(f"WARNING: {name} has non-numeric columns: {non_numeric_cols}")
            # Print the first few rows of problematic columns
            print(df[non_numeric_cols].head())
        return df.to_numpy(dtype=np.float32)
    
    clinical = check_numeric(clinical, "Clinical")
    mrna = check_numeric(mrna, "mRNA")
    mutation = check_numeric(mutation, "Mutation")

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
    y = y.squeeze() # converts from [x, 1] to [x] shape

    ds = TensorDataset(
        torch.tensor(clinical, dtype=torch.float32),
        torch.tensor(mrna, dtype=torch.float32),
        torch.tensor(mutation, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)


# =======================
# Neural Network
# =======================
class SimpleMultimodalNet(nn.Module):
    def __init__(self, clin_dim, mrna_dim, mut_dim, hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT, lr=config.LEARNING_RATE):
        super().__init__()

        # Separate encoders for each modality
        self.clinical_fc = nn.Sequential(
            nn.Linear(clin_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        self.mrna_fc = nn.Sequential(
            nn.Linear(mrna_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        self.mut_fc = nn.Sequential(
            nn.Linear(mut_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary output
        )
        
    def forward(self, clin, mrna, mut):
        clin_emb = self.clinical_fc(clin)
        mrna_emb = self.mrna_fc(mrna)
        mut_emb = self.mut_fc(mut)
        
        # Concatenate embeddings
        fused = torch.cat([clin_emb, mrna_emb, mut_emb], dim=1)
        output = self.fusion_fc(fused)
        return output.squeeze()


def train_one_fold(train_loader, val_loader, model, optimizer, criterion, seed, report_to_ray=True):
    set_random_seed(seed, deterministic=True)

    best_val_auroc, best_state, patience_counter = 0, None, 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for clin, mrna, mut, y in train_loader:
            clin, mrna, mut, y = clin.to(device), mrna.to(device), mut.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(clin, mrna, mut)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for clin, mrna, mut, y in val_loader:
                clin, mrna, mut, y = clin.to(device), mrna.to(device), mut.to(device), y.to(device)
                probs = torch.sigmoid(model(clin, mrna, mut)).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        val_auroc = roc_auc_score(all_labels, all_probs)

        if report_to_ray:
            tune.report(val_auroc=val_auroc)
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

    return best_state

def train_ray_trial(config):
    fold = config["fold"]
    clinical_df = config["clinical_df"]
    mrna_df = config["mrna_df"]
    mutation_df = config["mutation_df"]
    labels = config["labels"]
    train_idx = config["train_idx"]
    val_idx = config["val_idx"]

def evaluate_with_threshold(model, loader, threshold):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for clin, mrna, mut, y in loader:
            clin, mrna, mut, y = clin.to(device), mrna.to(device), mut.to(device), y.to(device)
            probs = torch.sigmoid(model(clin, mrna, mut)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs >= threshold).astype(float)

    metrics = {
        'auroc': roc_auc_score(all_labels, all_probs),
        'auprc': average_precision_score(all_labels, all_probs),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'f1': f1_score(all_labels, preds, zero_division=0)
    }
    return metrics


def run_kfold_gridsearch_with_preprocessing(
    clinical_df, mrna_df, mutation_df, labels,
    param_grid,
    k=5,
):
    """
    Grid search with repeated stratified K-fold CV, fitting preprocessors and
    feature selectors (SelectFromModel) within each fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.SEED)
    folds = list(skf.split(np.arange(len(labels)), labels))

    # indices = np.arange(len(labels))
    # param_combinations = list(product(*param_grid.values()))
    # best_hyperparams, best_score = None, -1
    # results_summary = {}

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\n===== Hyperparams: {param_dict} =====")

        fold_metrics = {'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'f1': []}

        for fold, (train_idx, val_idx) in enumerate(rskf.split(indices, labels)):
            print(f"\n--- Fold {fold + 1}/{k} ---")

            clin_train, clin_val = clinical_df.iloc[train_idx], clinical_df.iloc[val_idx]
            mrna_train, mrna_val = mrna_df.iloc[train_idx], mrna_df.iloc[val_idx]
            mut_train, mut_val = mutation_df.iloc[train_idx], mutation_df.iloc[val_idx]
            y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

            clinical_prep = ClinicalPreprocessorWrapper(
                cols_to_remove=config.CLINICAL_COLS_TO_REMOVE,
                categorical_cols=config.CATEGORICAL_COLS,
                max_null_frac=config.CLINICAL_MAX_NULL_FRAC,
                uniform_thresh=config.CLINICAL_UNIFORM_THRESH,
            )
            mrna_prep = MrnaPreprocessorWrapper(
                max_null_frac=config.MAX_NULL_FRAC,
                uniform_thresh=config.UNIFORM_THRESHOLD,
                random_state=config.SEED,
            )
            mutation_prep = MutationPreprocessorWrapper(
                max_mutation_count=param_dict["max_mutation_count"],
                uniform_thresh=param_dict["mutation_uniform_thresh"],
            )

            clinical_prep.fit(clin_train)
            mrna_prep.fit(mrna_train, y_train)
            mutation_prep.fit(mut_train)

            clin_train = clinical_prep.transform(clin_train)
            clin_val = clinical_prep.transform(clin_val)

            mrna_train = mrna_prep.transform(mrna_train)
            mrna_val = mrna_prep.transform(mrna_val)

            mut_train = mutation_prep.transform(mut_train)
            mut_val = mutation_prep.transform(mut_val)

            # === SelectFromModel for mRNA ===
            mrna_model_cls = param_dict["mrna_model"]  # e.g., LogisticRegression, RandomForestClassifier
            mrna_model_params = param_dict.get("mrna_model_params", {})  # estimator hyperparameters
            
            # If using XGBClassifier, inject safe/effective defaults for SelectFromModel
            if mrna_model_cls is XGBClassifier:
                # ensure importance_type is gain so feature_importances_ is meaningful
                mrna_model_params = {
                    **mrna_model_params,
                    "importance_type": "gain",
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "random_state": config.SEED,
                }
            
            sfm_mrna = SelectFromModel(
                estimator=mrna_model_cls(**mrna_model_params),
                threshold=param_dict.get("mrna_threshold", "median"),
                max_features=param_dict.get("mrna_max_features", None)
            )
            
            sfm_mrna.fit(mrna_train, y_train)
            mrna_train = sfm_mrna.transform(mrna_train)
            mrna_val   = sfm_mrna.transform(mrna_val)
            
            
            # === SelectFromModel for mutation ===
            mut_model_cls = param_dict["mut_model"]
            mut_model_params = param_dict.get("mut_model_params", {})
            
            if mut_model_cls is XGBClassifier:
                mut_model_params = {
                    **mut_model_params,
                    "importance_type": "gain",
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "random_state": config.SEED,
                }
            
            sfm_mut = SelectFromModel(
                estimator=mut_model_cls(**mut_model_params),
                threshold=param_dict.get("mut_threshold", "median"),
                max_features=param_dict.get("mut_max_features", None)
            )
            
            sfm_mut.fit(mut_train, y_train)
            mut_train = sfm_mut.transform(mut_train)
            mut_val   = sfm_mut.transform(mut_val)

            # --- Build datasets and dataloaders ---
            train_loader = to_loader(clin_train, mrna_train, mut_train, y_train, shuffle=True)
            val_loader = to_loader(clin_val, mrna_val, mut_val, y_val)

            model = SimpleMultimodalNet(clin_train.shape[1], mrna_train.shape[1], mut_train.shape[1],
                                        param_dict["hidden_dim"], param_dict["dropout"], param_dict["lr"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=param_dict.get('lr', 1e-3))
            pos_weight_value = torch.tensor((len(y_train) - y_train.sum()) / y_train.sum(), dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

            best_state = train_one_fold(train_loader, val_loader, model, optimizer, criterion, config.SEED + fold)
            model.load_state_dict(best_state)

            # Threshold tuning, evaluation identical
            all_probs, all_labels = [], []
            with torch.no_grad():
                for clin, mrna, mut, y in val_loader:
                    clin, mrna, mut, y = clin.to(device), mrna.to(device), mut.to(device), y.to(device)
                    probs = torch.sigmoid(model(clin, mrna, mut)).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(y.cpu().numpy())
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)

            thresholds = np.linspace(0.001, 0.9, 200)
            best_t, best_metric = 0.5, -1
            for t in thresholds:
                preds = (all_probs >= t).astype(float)
                score = f1_score(all_labels, preds, zero_division=0)
                if score > best_metric:
                    best_metric, best_t = score, t

            val_metrics = evaluate_with_threshold(model, val_loader, best_t)

            print(f"Val metrics: {val_metrics}")

            for k_ in fold_metrics.keys():
                fold_metrics[k_].append(val_metrics[k_])

        results_summary[str(param_dict)] ={k: (mean(v), stdev(v)) for k, v in fold_metrics.items()}

        mean_f1 = results_summary[str(param_dict)]["f1"][0]
        if mean_f1 > best_score:
            best_score = mean_f1
            best_hyperparams = param_dict

    print(f"\n=== Best hyperparameters: {best_hyperparams} (mean F1 = {best_score:.4f}) ===")
    return results_summary, best_hyperparams

#### XGBoost PARAM_GRID ###################################################################
param_grid = {
    # Models for mRNA
    "mrna_model": [
        XGBClassifier,
    ],

    # Hyperparameters for each mRNA model (small set so it runs quickly)
    "mrna_model_params": [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9},
    ],

    # Thresholds for mRNA selection
    "mrna_threshold": ["median"],
    "mrna_max_features": [None],

    # Models for mutation
    "mut_model": [
        XGBClassifier,
    ],

    # Hyperparameters for each mutation model
    "mut_model_params": [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9},
    ],

    # Thresholds for mutation selection
    "mut_threshold": ["median"],
    "mut_max_features": [None],

    # NN hyperparameters
    "hidden_dim": [64],
    "dropout": [0.2],
    "lr": [1e-3],
    "max_mutation_count": [1],
    "mutation_uniform_thresh": [0.98],
}

results_summary, best_hyperparams = run_kfold_gridsearch_with_preprocessing(
    clinical_train, mrna_train, mutation_train, y_train,
    clinical_test, mrna_test, mutation_test, y_test,
    param_grid,
    k=3,
    n_repeats=1,
    optimize_metric='f1'
)

def print_results_summary(results_summary):
    """
    Prints the results summary from the k-fold grid search.
    Also prints which hyperparameter set achieved:
      - Best internal AUROC
      - Best internal AUPRC
      - Best external AUROC
      - Best external AUPRC
    """

    # ---- Print the full summary exactly as before ----
    for param_str, metrics in results_summary.items():
        print("="*60)
        print(f"Hyperparameters: {param_str}")
        print("-"*60)
        for phase in ["internal", "external"]:
            print(f"{phase.upper()} METRICS:")
            for metric, (mean_val, std_val) in metrics[phase].items():
                print(f"  {metric:10}: {mean_val:.4f} ± {std_val:.4f}")
        print("="*60 + "\n")

    # ---- Initialize best trackers ----
    best_internal_auc = best_internal_auprc = None
    best_external_auc = best_external_auprc = None

    best_internal_auc_params = best_internal_auprc_params = None
    best_external_auc_params = best_external_auprc_params = None

    # ---- Compute best metrics ----
    for param_str, metrics in results_summary.items():

        int_auc = metrics["internal"]["auroc"][0]
        int_auprc = metrics["internal"]["auprc"][0]
        ext_auc = metrics["external"]["auroc"][0]
        ext_auprc = metrics["external"]["auprc"][0]

        # Internal AUROC
        if (best_internal_auc is None) or (int_auc > best_internal_auc):
            best_internal_auc = int_auc
            best_internal_auc_params = param_str

        # Internal AUPRC
        if (best_internal_auprc is None) or (int_auprc > best_internal_auprc):
            best_internal_auprc = int_auprc
            best_internal_auprc_params = param_str

        # External AUROC
        if (best_external_auc is None) or (ext_auc > best_external_auc):
            best_external_auc = ext_auc
            best_external_auc_params = param_str

        # External AUPRC
        if (best_external_auprc is None) or (ext_auprc > best_external_auprc):
            best_external_auprc = ext_auprc
            best_external_auprc_params = param_str

    # ---- Print the best results ----
    print("\n" + "#"*60)
    print(" BEST MODELS — INTERNAL VALIDATION")
    print("#"*60)

    print(f"\nHighest INTERNAL AUROC: {best_internal_auc:.4f}")
    print(f"Hyperparameters: {best_internal_auc_params}")

    print(f"\nHighest INTERNAL AUPRC: {best_internal_auprc:.4f}")
    print(f"Hyperparameters: {best_internal_auprc_params}")

    print("\n" + "#"*60)
    print(" BEST MODELS — EXTERNAL VALIDATION")
    print("#"*60)

    print(f"\nHighest EXTERNAL AUROC: {best_external_auc:.4f}")
    print(f"Hyperparameters: {best_external_auc_params}")

    print(f"\nHighest EXTERNAL AUPRC: {best_external_auprc:.4f}")
    print(f"Hyperparameters: {best_external_auprc_params}")

    print("#"*60 + "\n")

print_results_summary(results_summary)
joblib.dump(results_summary, "results_summary_xgb.pkl")
