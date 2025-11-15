import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config
import random
import numpy as np

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
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (patience={config.PATIENCE})")
                break

    return best_state


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

# =======================
# Neural Network
# =======================
class MultimodalNet(nn.Module):
    def __init__(self, 
                 clin_dim, mrna_dim, mut_dim, 
                 hidden_dim=config.HIDDEN_DIM, 
                 dropout=config.DROPOUT, 
                 lr=config.LEARNING_RATE,
                 use_gene_sel=True,
                ):
        super().__init__()
        
        self.use_gene_sel = use_gene_sel
        if use_gene_sel:
            # ===== Gene selectors (for omics modalities) =====
            self.gene_sel_mrna = GeneSelector(mrna_dim)
            self.gene_sel_mut = GeneSelector(mut_dim)

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
        if self.use_gene_sel:
            # Apply gene selectors
            mrna = self.gene_sel_mrna(mrna)
            mut = self.gene_sel_mut(mut)

        # Encode each modality
        clin_emb = self.clinical_fc(clin)
        mrna_emb = self.mrna_fc(mrna)
        mut_emb = self.mut_fc(mut)
        
        # Concatenate embeddings
        fused = torch.cat([clin_emb, mrna_emb, mut_emb], dim=1)
        output = self.fusion_fc(fused)
        return output.squeeze()