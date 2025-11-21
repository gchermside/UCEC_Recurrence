# Imports
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import joblib
import random
import pickle
import numpy as np
from statistics import mean, stdev
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from statistics import mean, stdev
import numpy as np
import torch, random, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config
from preprocessing_utils import *
from model_utils import *

# Set seeds for reproducibility
set_random_seed(config.SEED, deterministic=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalDataset(Dataset):
    def __init__(self, X, y, clinical_cols, mrna_cols, mutation_cols):
        # Split modalities
        self.clinical = X[clinical_cols].to_numpy()
        self.mrna = X[mrna_cols].to_numpy().astype(float)
        self.mutation = X[mutation_cols].to_numpy().astype(float)
        self.labels = y.to_numpy().astype(float)
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.clinical[idx], dtype=torch.float32),
            torch.tensor(self.mrna[idx], dtype=torch.float32),
            torch.tensor(self.mutation[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
X_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_val.joblib"))
y_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_val.joblib"))
X_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_test.joblib"))
y_test = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_test.joblib"))
clinical_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "clinical_cols.joblib"))
mrna_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mrna_cols.joblib"))
mutation_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mutation_cols.joblib"))

X_testval = pd.concat([X_val, X_test], axis=0).reset_index(drop=True)
y_testval = pd.concat([y_val, y_test], axis=0).reset_index(drop=True)

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

clinical_testval = X_testval[clinical_cols]
mrna_testval = X_testval[mrna_cols]
mutation_testval = X_testval[mutation_cols]

def to_loader(c, m, mu, y, shuffle=False):
    
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
    
    c = check_numeric(c, "Clinical")
    m = check_numeric(m, "mRNA")
    mu = check_numeric(mu, "Mutation")

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
    y = y.squeeze() # converts from [x, 1] to [x] shape

    ds = TensorDataset(
        torch.tensor(c),
        torch.tensor(m),
        torch.tensor(mu),
        torch.tensor(y)
    )
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)


# =======================
# Neural Network
# =======================
class MultimodalNet(nn.Module):
    def __init__(self,
                 clin_dim, mrna_dim, mut_dim,
                 clin_hidden=[64, 32],
                 mrna_hidden=[64, 32],
                 mut_hidden=[64, 32],
                 clin_dropout=[0, 0],
                 mrna_dropout=[0, 0],
                 mut_dropout=[0, 0],
                 activation='relu',
                 fusion_hidden=64,
                 fusion_dropout=0,
                 use_gene_sel=True,
                ):
        super().__init__()
        
        self.use_gene_sel = use_gene_sel
        if use_gene_sel:
            # ===== Gene selectors (for omics modalities) =====
            self.gene_sel_mrna = GeneSelector(mrna_dim)
            self.gene_sel_mut = GeneSelector(mut_dim)

        # ===== Modality encoders =====
        self.enc_clin = ModalityEncoder(clin_dim, clin_hidden, clin_dropout, activation)
        self.enc_mrna = ModalityEncoder(mrna_dim, mrna_hidden, mrna_dropout, activation)
        self.enc_mut  = ModalityEncoder(mut_dim, mut_hidden, mut_dropout, activation)


        # ===== Fusion + final classifier =====
        total_dim = clin_hidden[-1] + mrna_hidden[-1] + mut_hidden[-1]
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, 1)
        )
        
    def forward(self, clin, mrna, mut):
        if self.use_gene_sel:
            # Apply gene selectors
            mrna = self.gene_sel_mrna(mrna)
            mut = self.gene_sel_mut(mut)

        # Encode each modality
        clin_emb = self.enc_clin(clin)
        mrna_emb = self.enc_mrna(mrna)
        mut_emb  = self.enc_mut(mut)
        
        # Fuse and classify
        fused = torch.cat([clin_emb, mrna_emb, mut_emb], dim=1)
        output = self.fusion_fc(fused)
        return output.squeeze()


def train_one_fold(train_loader, val_loader, model, optimizer, criterion, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

    return best_state

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
    external_clinical_df, external_mrna_df, external_mutation_df, external_labels,
    param_grid,
    k=5,
    n_repeats=3,
    optimize_metric='f1',
):
    """
    Grid search with repeated stratified K-fold CV, fitting preprocessors and
    feature selectors (SelectFromModel) within each fold.
    """

    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=config.SEED)
    indices = np.arange(len(labels))
    param_combinations = list(product(*param_grid.values()))
    best_hyperparams, best_score = None, -1
    results_summary = {}

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\n===== Hyperparams: {param_dict} =====")

        fold_metrics = {'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'f1': []}
        external_metrics = {'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'f1': []}

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
            clin_ext = clinical_prep.transform(external_clinical_df.copy())

            mrna_train = mrna_prep.transform(mrna_train)
            mrna_val = mrna_prep.transform(mrna_val)
            mrna_ext = mrna_prep.transform(external_mrna_df.copy())

            mut_train = mutation_prep.transform(mut_train)
            mut_val = mutation_prep.transform(mut_val)
            mut_ext = mutation_prep.transform(external_mutation_df.copy())

            # === SelectFromModel for mRNA ===
            mrna_model_cls = param_dict["mrna_model"]  # e.g., LogisticRegression, RandomForestClassifier
            mrna_model_params = param_dict.get("mrna_model_params", {})  # estimator hyperparameters
            
            sfm_mrna = SelectFromModel(
                estimator=mrna_model_cls(**mrna_model_params),
                threshold=param_dict.get("mrna_threshold", "median"),
                max_features=param_dict.get("mrna_max_features", None)
            )
            
            sfm_mrna.fit(mrna_train, y_train)
            mrna_train = sfm_mrna.transform(mrna_train)
            mrna_val   = sfm_mrna.transform(mrna_val)
            mrna_ext   = sfm_mrna.transform(mrna_ext)
            
            
            # === SelectFromModel for mutation ===
            mut_model_cls = param_dict["mut_model"]
            mut_model_params = param_dict.get("mut_model_params", {})
            
            sfm_mut = SelectFromModel(
                estimator=mut_model_cls(**mut_model_params),
                threshold=param_dict.get("mut_threshold", "median"),
                max_features=param_dict.get("mut_max_features", None)
            )
            
            sfm_mut.fit(mut_train, y_train)
            mut_train = sfm_mut.transform(mut_train)
            mut_val   = sfm_mut.transform(mut_val)
            mut_ext   = sfm_mut.transform(mut_ext)

            # --- Build datasets and dataloaders ---
            train_loader = to_loader(clin_train, mrna_train, mut_train, y_train, shuffle=True)
            val_loader = to_loader(clin_val, mrna_val, mut_val, y_val)
            ext_loader = to_loader(clin_ext, mrna_ext, mut_ext, external_labels)

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
            ext_metrics = evaluate_with_threshold(model, ext_loader, best_t)

            print(f"Val metrics: {val_metrics}")
            print(f"Ext metrics: {ext_metrics}")

            for k_ in fold_metrics.keys():
                fold_metrics[k_].append(val_metrics[k_])
                external_metrics[k_].append(ext_metrics[k_])

        results_summary[str(param_dict)] = {
            "internal": {k: (mean(v), stdev(v)) for k, v in fold_metrics.items()},
            "external": {k: (mean(v), stdev(v)) for k, v in external_metrics.items()}
        }

        mean_f1 = results_summary[str(param_dict)]["internal"]["f1"][0]
        if mean_f1 > best_score:
            best_score = mean_f1
            best_hyperparams = param_dict

    print(f"\n=== Best hyperparameters: {best_hyperparams} (mean F1 = {best_score:.4f}) ===")
    return results_summary, best_hyperparams

#### Random Forest PARAM_GRID ###################################################################
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    # Models for mRNA
    "mrna_model": [
        RandomForestClassifier,
    ],

    # Hyperparameters for mRNA model (reduced to 1 stable config)
    "mrna_model_params": [
        {"n_estimators": 300, "max_depth": 10},
    ],

    # Thresholds for mRNA selection (reduced)
    "mrna_threshold": ["median", "mean"],
    "mrna_max_features": [None, 150],  # reduced from [None, 100, 200]

    # Models for mutation
    "mut_model": [
        RandomForestClassifier,
    ],

    # Hyperparameters for mutation model (reduced to 1)
    "mut_model_params": [
        {"n_estimators": 300, "max_depth": 10},
    ],

    # Thresholds for mutation selection (reduced)
    "mut_threshold": ["median", "mean"],
    "mut_max_features": [None, 75],  # reduced from [None, 50, 100]

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
joblib.dump(results_summary, "results_summary_random_forest.pkl")
