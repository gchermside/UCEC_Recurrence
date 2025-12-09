import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config
import random
import numpy as np
import pandas as pd
import copy

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
from statistics import mean, stdev
import numpy as np
import torch, random, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config

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

def set_random_seed(seed=config.SEED, deterministic=False):
    """ Sets random seed for reproducibility across random, numpy, torch (CPU and GPU) """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic: # if True, forces PyTorch to use deterministic algorithms (slower, more reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_with_threshold(model, loader, threshold, device):
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


def train_one_fold(train_loader, val_loader, model, optimizer, criterion, seed, device, verbose=True):
    """
    Trains the model for one fold with early stopping based on validation AUROC.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: nn.Module to train
        optimizer: optimizer instance
        criterion: loss function
        seed: random seed for reproducibility
        verbose: if True, print metrics each epoch (slower)
    Returns:
        best_state_dict: the model state dict from the best epoch
    """
    best_val_auroc, best_state, patience_counter = 0, None, 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_losses = [] if verbose else None  # Only track if needed

        for clin, mrna, mut, y in train_loader:
            clin, mrna, mut, y = clin.to(device), mrna.to(device), mut.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(clin, mrna, mut)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if verbose:
                train_losses.append(loss.item())

        # Validation
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

        # --- Print detailed metrics only if verbose ---
        if verbose:
            avg_train_loss = np.mean(train_losses)
            preds = (all_probs >= 0.5).astype(float)
            val_precision = precision_score(all_labels, preds, zero_division=0)
            val_recall = recall_score(all_labels, preds, zero_division=0)
            val_f1 = f1_score(all_labels, preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val AUROC: {val_auroc:.4f} | "
                  f"P: {val_precision:.3f} | R: {val_recall:.3f} | F1: {val_f1:.3f} | "
                  f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}"
            )

        # --- Early stopping ---
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (patience={config.PATIENCE})")
                break
    print("BEST best_val_auroc IS:", best_val_auroc)
    return best_state

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


def build_optimizer(model, config_params):
    """
    Create and return a PyTorch optimizer based on config_params.
    """
    optimizer_name = config_params.get("optimizer", "Adam")
    lr = config_params["lr"]
    weight_decay = config_params.get("weight_decay", 0.0)

    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config_params.get("weight_decay", 0.01)
        )

    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config_params.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=config_params.get("nesterov", True)
        )

    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=config_params.get("rmsprop_alpha", 0.99),
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def coral_loss_fn(source, target):
    """ CORAL loss between source and target feature tensors """
    d = source.size(1)
    s_c = source - source.mean(dim=0, keepdim=True) # center features
    s_cov = (s_c.T @ s_c) / (source.size(0) - 1) # compute covariance
    t_c = target - target.mean(dim=0, keepdim=True)
    t_cov = (t_c.T @ t_c) / (target.size(0) - 1)
    return ((s_cov - t_cov).pow(2).sum()) / (4 * d * d) # Frobenius norm between covariance matrices

class ModalityEncoder(nn.Module):    
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation):
        super(ModalityEncoder, self).__init__()
        layers = []
        in_dim = input_dim
        if dropout_rates is None or len(dropout_rates) == 0:
            dropout_rates = [0.0] * len(hidden_dims)
        elif len(dropout_rates) < len(hidden_dims):
            # Repeat last value
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(hidden_dims) - len(dropout_rates))
        elif len(dropout_rates) > len(hidden_dims):
            # Truncate
            dropout_rates = dropout_rates[:len(hidden_dims)]
        for h_dim, p in zip(hidden_dims, dropout_rates):
            layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim)])
            act = {'leaky_relu': nn.LeakyReLU, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]()
            if p == 0: layers.append(act)
            else: layers.extend([act, nn.Dropout(p)])
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    
class GeneSelector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
    def forward(self, x):
        return x * torch.sigmoid(self.weights) # x * self.weights # Genes with |weight| > 2σ retained

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
