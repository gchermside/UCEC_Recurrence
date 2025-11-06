import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
