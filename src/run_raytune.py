#!/usr/bin/env python3
"""
Ray Tune Hyperparameter Search with Feature Selection
Usage: python run_raytune.py <feature_selection_method>
Options: LogisticRegression, RandomForest, XGBoost, NoFeatureSelection
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import random
import pickle
import numpy as np
import pandas as pd
import tempfile
import argparse
from datetime import datetime

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import config
from preprocessing_utils import ClinicalPreprocessorWrapper, MrnaPreprocessorWrapper, MutationPreprocessorWrapper
from model_utils import GeneSelector, ModalityEncoder, set_random_seed

    
# Set seeds for reproducibility
set_random_seed(seed=config.SEED, deterministic=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def train_with_raytune(config_params):
    """
    Ray Tune trainable function for hyperparameter optimization.
    Includes optional feature selection before training.
    """
    # Import inside function for Ray workers
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import pandas as pd
    import random
    import ray
    from ray import tune
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    import config
    from model_utils import GeneSelector, ModalityEncoder
    from preprocessing_utils import ClinicalPreprocessorWrapper, MrnaPreprocessorWrapper, MutationPreprocessorWrapper
    import xgboost as xgb

    # Define MultimodalNet inside trainable
    class MultimodalNet(nn.Module):
        def __init__(self,
                     clin_dim, mrna_dim, mut_dim,
                     clin_hidden=config.CLIN_HIDDEN,
                     mrna_hidden=config.MRNA_HIDDEN,
                     mut_hidden=config.MUT_HIDDEN,
                     clin_dropout=config.CLIN_DROPOUT,
                     mrna_dropout=config.MRNA_DROPOUT,
                     mut_dropout=config.MUT_DROPOUT,
                     activation=config.ACTIVATION_FUNC,
                     fusion_hidden=config.FUSION_HIDDEN,
                     fusion_dropout=config.FUSION_DROPOUT,
                     use_gene_sel=True,
                    ):
            super().__init__()

            self.use_gene_sel = use_gene_sel
            if use_gene_sel:
                self.gene_sel_mrna = GeneSelector(mrna_dim)
                self.gene_sel_mut = GeneSelector(mut_dim)

            self.enc_clin = ModalityEncoder(clin_dim, clin_hidden, clin_dropout, activation)
            self.enc_mrna = ModalityEncoder(mrna_dim, mrna_hidden, mrna_dropout, activation)
            self.enc_mut  = ModalityEncoder(mut_dim, mut_hidden, mut_dropout, activation)

            total_dim = clin_hidden[-1] + mrna_hidden[-1] + mut_hidden[-1]
            self.fusion_fc = nn.Sequential(
                nn.Linear(total_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, 1)
            )

        def forward(self, clin, mrna, mut):
            if self.use_gene_sel:
                mrna = self.gene_sel_mrna(mrna)
                mut = self.gene_sel_mut(mut)

            clin_emb = self.enc_clin(clin)
            mrna_emb = self.enc_mrna(mrna)
            mut_emb  = self.enc_mut(mut)

            fused = torch.cat([clin_emb, mrna_emb, mut_emb], dim=1)
            output = self.fusion_fc(fused)
            return output.squeeze()

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
            return df.to_numpy(dtype=np.float32)

        c = check_numeric(c, "Clinical")
        m = check_numeric(m, "mRNA")
        mu = check_numeric(mu, "Mutation")

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            y = np.array(y, dtype=np.float32).reshape(-1, 1)
        y = y.squeeze()

        ds = TensorDataset(
            torch.tensor(c),
            torch.tensor(m),
            torch.tensor(mu),
            torch.tensor(y)
        )
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    # Retrieve data from Ray's object store
    clin_train = ray.get(config_params["clinical_train_ref"])
    mrna_train = ray.get(config_params["mrna_train_ref"])
    mut_train = ray.get(config_params["mutation_train_ref"])
    y_train = ray.get(config_params["y_train_ref"])

    clin_val = ray.get(config_params["clinical_val_ref"])
    mrna_val = ray.get(config_params["mrna_val_ref"])
    mut_val = ray.get(config_params["mutation_val_ref"])
    y_val = ray.get(config_params["y_val_ref"])

    # Set seed
    seed = config_params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== PREPROCESSING =====
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
        max_mutation_count=config_params.get("max_mutation_count", config.MAX_MUTATION_COUNT), #FIXME: tune these!
        uniform_thresh=config_params.get("mutation_uniform_thresh", config.MUTATION_UNIFORM_THRESH),
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

    # ===== FEATURE SELECTION =====
    fs_method = config_params.get("fs_method", "NoFeatureSelection")

    def apply_feature_selection(estimator, mrna_train, mrna_val, mut_train, mut_val, y_train):
        """Helper function to apply SelectFromModel to both mRNA and mutation data"""
        threshold = config_params.get("fs_threshold", "median")
        max_features = config_params.get("fs_max_features", None)

        # Create and fit SelectFromModel for mRNA
        sfm_mrna = SelectFromModel(estimator, threshold=threshold, max_features=max_features)
        sfm_mrna.fit(mrna_train, y_train)

        # Create and fit SelectFromModel for mutations
        sfm_mut = SelectFromModel(estimator, threshold=threshold, max_features=max_features)
        sfm_mut.fit(mut_train, y_train)

        # Transform data
        mrna_train_fs = sfm_mrna.transform(mrna_train)
        mrna_val_fs = sfm_mrna.transform(mrna_val)
        mut_train_fs = sfm_mut.transform(mut_train)
        mut_val_fs = sfm_mut.transform(mut_val)

        return mrna_train_fs, mrna_val_fs, mut_train_fs, mut_val_fs

    # Create estimator based on feature selection method
    if fs_method == "LogisticRegression":
        fs_estimator = LogisticRegression(
            C=config_params.get("fs_C", 1.0),
            penalty=config_params.get("fs_penalty", "l2"),
            max_iter=1000,
            solver='saga',
            random_state=config.SEED
        )
        mrna_train, mrna_val, mut_train, mut_val = apply_feature_selection(
            fs_estimator, mrna_train, mrna_val, mut_train, mut_val, y_train
        )

    elif fs_method == "RandomForest":
        fs_estimator = RandomForestClassifier(
            n_estimators=config_params.get("fs_n_estimators", 100),
            max_depth=config_params.get("fs_max_depth", None),
            random_state=config.SEED
        )
        mrna_train, mrna_val, mut_train, mut_val = apply_feature_selection(
            fs_estimator, mrna_train, mrna_val, mut_train, mut_val, y_train
        )

    elif fs_method == "XGBoost":
        if xgb is None:
            raise ImportError("XGBoost not available")
        fs_estimator = xgb.XGBClassifier(
            learning_rate=config_params.get("fs_learning_rate", 0.1),
            max_depth=config_params.get("fs_max_depth", 6),
            n_estimators=config_params.get("fs_n_estimators", 100),
            random_state=config.SEED,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        mrna_train, mrna_val, mut_train, mut_val = apply_feature_selection(
            fs_estimator, mrna_train, mrna_val, mut_train, mut_val, y_train
        )

    # NoFeatureSelection: skip feature selection (no transformation needed)

    # Create data loaders with preprocessed (and possibly feature-selected) data
    train_loader = to_loader(clin_train, mrna_train, mut_train, y_train, shuffle=True)
    val_loader = to_loader(clin_val, mrna_val, mut_val, y_val)

    # Get dimensions after preprocessing/feature selection
    clin_dim = clin_train.shape[1]
    mrna_dim = mrna_train.shape[1]
    mut_dim = mut_train.shape[1]

    # Create model
    model = MultimodalNet(
        clin_dim=clin_dim,
        mrna_dim=mrna_dim,
        mut_dim=mut_dim,
        clin_hidden=config_params["clin_hidden"],
        mrna_hidden=config_params["mrna_hidden"],
        mut_hidden=config_params["mut_hidden"],
        clin_dropout=config_params["clin_dropout"],
        mrna_dropout=config_params["mrna_dropout"],
        mut_dropout=config_params["mut_dropout"],
        activation=config_params.get("activation", "relu"),
        fusion_hidden=config_params["fusion_hidden"],
        fusion_dropout=config_params["fusion_dropout"],
        use_gene_sel=False  # No gene selection in model
    ).to(device)

    # Optimizer and loss
    optimizer_name = config_params.get("optimizer", "Adam")
    lr = config_params["lr"]

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config_params.get("weight_decay", 0.0)
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config_params.get("weight_decay", 0.01)
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config_params.get("momentum", 0.9),
            weight_decay=config_params.get("weight_decay", 0.0),
            nesterov=config_params.get("nesterov", True)
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=config_params.get("rmsprop_alpha", 0.99),
            weight_decay=config_params.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    pos_weight_value = torch.tensor(
        (len(y_train) - y_train.sum()) / y_train.sum(),
        dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

    # Training loop
    best_val_auroc = 0
    patience_counter = 0
    patience = config_params["patience"]
    num_epochs = config_params["num_epochs"]

    for epoch in range(num_epochs):
        # Training
        model.train()
        for clin, mrna, mut, y in train_loader:
            clin = clin.to(device)
            mrna = mrna.to(device)
            mut = mut.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(clin, mrna, mut)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for clin, mrna, mut, y in val_loader:
                clin = clin.to(device)
                mrna = mrna.to(device)
                mut = mut.to(device)
                y = y.to(device)

                probs = torch.sigmoid(model(clin, mrna, mut)).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Calculate metrics
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_auprc = average_precision_score(all_labels, all_probs)

        # Find optimal threshold for F1
        thresholds = np.linspace(0.001, 0.9, 200)
        best_f1 = 0
        for t in thresholds:
            preds = (all_probs >= t).astype(float)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        # Report metrics to Ray Tune
        tune.report({
            "auroc": val_auroc,
            "auprc": val_auprc,
            "f1": best_f1,
            "epoch": epoch
        })

        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break


def get_search_space(fs_method, clinical_train_ref, mrna_train_ref, mutation_train_ref, y_train_ref,
                     clinical_val_ref, mrna_val_ref, mutation_val_ref, y_val_ref):
    """
    Returns the search space for Ray Tune based on the feature selection method.
    """
    # Base search space (same for all methods)
    base_space = {
        # Data references
        "clinical_train_ref": clinical_train_ref,
        "mrna_train_ref": mrna_train_ref,
        "mutation_train_ref": mutation_train_ref,
        "y_train_ref": y_train_ref,
        "clinical_val_ref": clinical_val_ref,
        "mrna_val_ref": mrna_val_ref,
        "mutation_val_ref": mutation_val_ref,
        "y_val_ref": y_val_ref,

        # Fixed parameters
        "seed": config.SEED,
        "patience": config.PATIENCE,
        "num_epochs": config.NUM_EPOCHS,

        # Feature selection method
        "fs_method": fs_method,

        # Neural network architecture hyperparameters
        "clin_hidden": tune.choice([
            [64],
            [128],
            [64, 32],
            [128, 64],
            [64, 32, 16],
        ]),
        "mrna_hidden": tune.choice([
            [64],
            [128],
            [128, 64],
            [256, 128],
            [128, 64, 32],
        ]),
        "mut_hidden": tune.choice([
            [64],
            [128],
            [64, 32],
            [128, 64],
            [64, 32, 16],
        ]),
        "clin_dropout": tune.choice([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.3, 0.3],
        ]),
        "mrna_dropout": tune.choice([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.3, 0.3],
        ]),
        "mut_dropout": tune.choice([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.3, 0.3],
        ]),
        "fusion_hidden": tune.choice([64, 128, 256]),
        "fusion_dropout": tune.uniform(0.0, 0.5),
        "lr": tune.loguniform(1e-5, 1e-2),
        "activation": tune.choice(["leaky_relu", "relu", "gelu", "silu"]),

        # Mutation preprocessing hyperparameters
        "max_mutation_count": tune.choice([5, 10, 15, 20]),
        "mutation_uniform_thresh": tune.uniform(0.90, 0.99),

        # Optimizer selection
        "optimizer": tune.choice(["Adam", "AdamW", "SGD", "RMSprop"]),

        # Optimizer-specific hyperparameters
        # weight_decay (for Adam, AdamW, SGD, RMSprop)
        "weight_decay": tune.loguniform(1e-6, 1e-2),

        # SGD-specific
        "momentum": tune.uniform(0.8, 0.99),  # Only used if optimizer=SGD
        "nesterov": tune.choice([True, False]),  # Only used if optimizer=SGD

        # RMSprop-specific
        "rmsprop_alpha": tune.uniform(0.9, 0.999),  # Only used if optimizer=RMSprop
    }

    # Add feature selection specific hyperparameters
    if fs_method == "LogisticRegression":
        base_space.update({
            "fs_C": tune.loguniform(0.001, 10.0),
            "fs_penalty": tune.choice(["l1", "l2"]),
            "fs_threshold": tune.choice(["mean", "median", "0.5*mean"]),
            "fs_max_features": tune.choice([None, 100, 200, 500]),
        })
    elif fs_method == "RandomForest":
        base_space.update({
            "fs_n_estimators": tune.choice([50, 100, 200]),
            "fs_max_depth": tune.choice([None, 5, 10, 20]),
            "fs_threshold": tune.choice(["mean", "median", "0.5*mean"]),
            "fs_max_features": tune.choice([None, 100, 200, 500]),
        })
    elif fs_method == "XGBoost":
        base_space.update({
            "fs_learning_rate": tune.loguniform(0.01, 0.3),
            "fs_max_depth": tune.choice([3, 5, 7, 9]),
            "fs_n_estimators": tune.choice([50, 100, 200]),
            "fs_threshold": tune.choice(["mean", "median", "0.5*mean"]),
            "fs_max_features": tune.choice([None, 100, 200, 500]),
        })
    # NoFeatureSelection: no additional parameters

    return base_space

def main():
    parser = argparse.ArgumentParser(description='Run Ray Tune hyperparameter search with feature selection')
    parser.add_argument('fs_method', type=str,
                       choices=['LogisticRegression', 'RandomForest', 'XGBoost', 'NoFeatureSelection'],
                       help='Feature selection method to use')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of hyperparameter combinations to try (default: 50)')
    parser.add_argument('--gpus_per_trial', type=float, default=1,
                       help='GPUs per trial (default: 1)')

    args = parser.parse_args()

    fs_method = args.fs_method
    num_samples = args.num_samples

    print(f"\n{'='*80}")
    print(f"Ray Tune Hyperparameter Search")
    print(f"Feature Selection Method: {fs_method}")
    print(f"Number of Trials: {num_samples}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    X_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_train.joblib"))
    y_train = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_train.joblib"))
    X_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "X_val.joblib"))
    y_val = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "y_val.joblib"))

    clinical_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "clinical_cols.joblib"))
    mrna_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mrna_cols.joblib"))
    mutation_cols = joblib.load(os.path.join(config.SPLIT_DATA_DIR, "mutation_cols.joblib"))

    # Split by modality
    clinical_train = X_train[clinical_cols]
    mrna_train = X_train[mrna_cols]
    mutation_train = X_train[mutation_cols]

    clinical_val = X_val[clinical_cols]
    mrna_val = X_val[mrna_cols]
    mutation_val = X_val[mutation_cols]

    print(f"Data loaded: {len(X_train)} train samples, {len(X_val)} validation samples")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Put data in Ray's object store
    print("Loading data into Ray object store...")
    clinical_train_ref = ray.put(clinical_train)
    mrna_train_ref = ray.put(mrna_train)
    mutation_train_ref = ray.put(mutation_train)
    y_train_ref = ray.put(y_train)

    clinical_val_ref = ray.put(clinical_val)
    mrna_val_ref = ray.put(mrna_val)
    mutation_val_ref = ray.put(mutation_val)
    y_val_ref = ray.put(y_val)

    # Get search space
    search_space = get_search_space(
        fs_method,
        clinical_train_ref, mrna_train_ref, mutation_train_ref, y_train_ref,
        clinical_val_ref, mrna_val_ref, mutation_val_ref, y_val_ref
    )

    # Configure scheduler and reporter
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=config.NUM_EPOCHS,
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["auroc", "auprc", "f1", "epoch"],
        max_report_frequency=30
    )

    # Resources per trial
    resources_per_trial = {
        "cpu": 1,
        "gpu": args.gpus_per_trial
    }

    print(f"\nStarting Ray Tune search...")
    print(f"Resources per trial: {resources_per_trial}")

    # Run Ray Tune
    result = tune.run(
        train_with_raytune,
        config=search_space,
        resources_per_trial=resources_per_trial,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=tempfile.mkdtemp(),
        name=f"raytune_{fs_method}",
        verbose=1,
        raise_on_failed_trial=False
    )

    # Get best trial
    best_trial = result.get_best_trial("f1", "max", "last")

    print(f"\n{'='*80}")
    print(f"Best trial results:")
    print(f"  F1: {best_trial.last_result['f1']:.4f}")
    print(f"  AUROC: {best_trial.last_result['auroc']:.4f}")
    print(f"  AUPRC: {best_trial.last_result['auprc']:.4f}")
    print(f"{'='*80}\n")

    # Extract hyperparameters (exclude data references and fixed params)
    best_hyperparams = {
        k: v for k, v in best_trial.config.items()
        if not k.endswith("_ref") and k not in ["seed", "patience", "num_epochs"]
    }

    # Save results
    output_prefix = f"{fs_method}_raytune"

    # Save best hyperparameters
    with open(f'{output_prefix}_best_hyperparams.pkl', 'wb') as f:
        pickle.dump(best_hyperparams, f)
    print(f"Saved best hyperparameters to '{output_prefix}_best_hyperparams.pkl'")

    # Save experiment results
    experiment_results = {
        "fs_method": fs_method,
        "best_hyperparams": best_hyperparams,
        "best_metrics": {
            "f1": best_trial.last_result['f1'],
            "auroc": best_trial.last_result['auroc'],
            "auprc": best_trial.last_result['auprc'],
        },
        "num_samples": num_samples,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "all_trials_df": result.dataframe(),
    }

    with open(f'{output_prefix}_results.pkl', 'wb') as f:
        pickle.dump(experiment_results, f)
    print(f"Saved experiment results to '{output_prefix}_results.pkl'")

    # Save trial dataframe as CSV for easy viewing
    result.dataframe().to_csv(f'{output_prefix}_all_trials.csv', index=False)
    print(f"Saved all trials to '{output_prefix}_all_trials.csv'")

    print(f"\n{'='*80}")
    print(f"Experiment completed successfully!")
    print(f"Feature Selection: {fs_method}")
    print(f"Best F1: {best_trial.last_result['f1']:.4f}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
