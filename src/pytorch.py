# Imports
import os
import sys
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
from statistics import mean, stdev
import numpy as np
import torch, random, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config
from preprocessing_utils import *

# --------------------------
# Set seeds for reproducibility
# --------------------------
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Simple Neural Network
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

def run_kfold_gridsearch_with_preprocessing(
    clinical_df, mrna_df, mutation_df, labels,
    external_clinical_df, external_mrna_df, external_mutation_df, external_labels,
    param_grid,
    k=5,
    n_repeats=3,
    optimize_metric='f1',
    feature_selection="none" # options are currently "none" and "stability_selection"
):
    """
    Grid search with repeated stratified K-fold CV, fitting preprocessors within each fold.
    Uses external validation set (transformed per fold) and adds AUPRC metric.
    """

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

            # --- Validation phase ---
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

    # --- Initialize folds ---
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=config.SEED)
    indices = np.arange(len(labels))
    param_combinations = list(product(*param_grid.values()))
    best_hyperparams, best_score = None, -1
    results_summary = {}

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        # print(f"\n===== Hyperparams: {param_dict} =====")

        fold_metrics = {'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'f1': []}
        external_metrics = {'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'f1': []}

        for fold, (train_idx, val_idx) in enumerate(rskf.split(indices, labels)):
            # print(f"\n--- Fold {fold + 1}/{k} ---")

            # Split data
            clin_train, clin_val = clinical_df.iloc[train_idx], clinical_df.iloc[val_idx]
            mrna_train, mrna_val = mrna_df.iloc[train_idx], mrna_df.iloc[val_idx]
            mut_train, mut_val = mutation_df.iloc[train_idx], mutation_df.iloc[val_idx]
            y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

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
                use_stability_selection=config.USE_STABILITY_SELECTION, 
                n_boots=config.N_BOOTS_FPR, # NOTE: might want to experiment with these values, they are set pretty strict right now and I'm not sure that is good for pytorch
                fpr_alpha=config.FPR_ALPHA, #FIXME: put back to config
                stability_threshold=config.STABILITY_THRESHOLD_FPR, # FIXME: put this back to config
                random_state=config.SEED,
            )
            mutation_prep = MutationPreprocessorWrapper(
                max_null_frac=config.MUTATION_MAX_NULL_FRAC,
                uniform_thresh=config.MUTATION_UNIFORM_THRESH,
            )

            clinical_prep.fit(clin_train)
            mrna_prep.fit(mrna_train, y_train)
            mutation_prep.fit(mut_train)

            # === Transform ===
            clin_train = clinical_prep.transform(clin_train)
            clin_val = clinical_prep.transform(clin_val)
            clin_ext = clinical_prep.transform(external_clinical_df.copy())

            mrna_train = mrna_prep.transform(mrna_train)
            mrna_val = mrna_prep.transform(mrna_val)
            mrna_ext = mrna_prep.transform(external_mrna_df.copy())

            mut_train = mutation_prep.transform(mut_train)
            mut_val = mutation_prep.transform(mut_val)
            mut_ext = mutation_prep.transform(external_mutation_df.copy())

            # --- Build datasets and dataloaders ---
            def to_loader(c, m, mu, y, shuffle=False):
                import pandas as pd
                import numpy as np
                import torch
                from torch.utils.data import TensorDataset, DataLoader
            
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
            if feature_selection == "stability_selection":
                # ===# --- Stability selection for mRNA ---
                print("Applying stability selection for mRNA...")
                mrna_ss_params = {
                    'n_boots': param_dict.get('mrna_n_boots', config.N_BOOTS_FPR),
                    'fpr_alpha': param_dict.get('mrna_fpr_alpha', config.FPR_ALPHA),
                    'stability_threshold': param_dict.get('mrna_stability_threshold', config.STABILITY_THRESHOLD_FPR),
                    'random_state': config.SEED
                }
                mrna_stability_selector = StabilitySelection(**mrna_ss_params)
                mrna_stability_selector.fit(mrna_train, y_train)
                mrna_train = mrna_stability_selector.transform(mrna_train)
                mrna_val = mrna_stability_selector.transform(mrna_val)
                mrna_ext = mrna_stability_selector.transform(mrna_ext)
                
                
                # --- Stability selection for Mutation ---
                mut_ss_params = {
                    'n_boots': param_dict.get('mut_n_boots', config.N_BOOTS_FPR),
                    'fpr_alpha': param_dict.get('mut_fpr_alpha', config.FPR_ALPHA),
                    'stability_threshold': param_dict.get('mut_stability_threshold', config.STABILITY_THRESHOLD_FPR),
                    'random_state': config.SEED
                }
                mut_stability_selector = StabilitySelection(**mut_ss_params)
                mut_stability_selector.fit(mut_train, y_train)
                mut_train = mut_stability_selector.transform(mut_train)
                mut_val = mut_stability_selector.transform(mut_val)
                mut_ext = mut_stability_selector.transform(mut_ext)

            train_loader = to_loader(clin_train, mrna_train, mut_train, y_train, shuffle=True)
            val_loader = to_loader(clin_val, mrna_val, mut_val, y_val)
            ext_loader = to_loader(clin_ext, mrna_ext, mut_ext, external_labels)

            # --- Train ---
            model = SimpleMultimodalNet(clin_train.shape[1], mrna_train.shape[1], mut_train.shape[1], param_dict["hidden_dim"], param_dict["dropout"], param_dict["lr"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=param_dict.get('lr', 1e-3))
            pos_weight_value = torch.tensor(
                (len(y_train) - y_train.sum()) / y_train.sum(),
                dtype=torch.float32
            ).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

            best_state = train_one_fold(train_loader, val_loader, model, optimizer, criterion, config.SEED + fold)
            model.load_state_dict(best_state)

            # --- Find ideal threshold on validation set ---
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

            # --- Evaluate validation and external sets ---
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


param_grid = {
    'dropout': [0],
    'hidden_dim': [32, 64, 128],
    'lr': [1e-3, 1e-4, 1e-5],
    # mRNA stability selection hyperparams
    'mrna_n_boots': [50],
    'mrna_fpr_alpha': [0.01, 0.05, 0.1],
    'mrna_stability_threshold': [0.65, 0.75, 0.85],
    # Mutation stability selection hyperparams
    'mut_n_boots': [50],
    'mut_fpr_alpha': [0.01, 0.05, 0.1],
    'mut_stability_threshold': [0.65, 0.75, 0.85]
}

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

if __name__ == "__main__":
    feature_selection = sys.argv[1] if len(sys.argv) > 1 else None
    
    results_summary, best_hyperparams = run_kfold_gridsearch_with_preprocessing(
        clinical_train, mrna_train, mutation_train, y_train,
        clinical_val, mrna_val, mutation_val, y_val,
        param_grid,
        k=3,
        n_repeats=1,
        optimize_metric='f1',
        feature_selection=feature_selection
    )
    
    print("BEST HYPERPARAMETERS", best_hyperparams)
    
    # Save results summary
    with open("gridsearch_results_summary_satability_selection.pkl", "wb") as f:
        pickle.dump(results_summary, f)
    
