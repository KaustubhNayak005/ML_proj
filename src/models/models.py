import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.gat import GATConv
from .layers.camouflage_gat import CamouflageGATConv
from .layers.sage import SAGEConv

class FraudGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out

class FraudCamouflageGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CamouflageGATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = CamouflageGATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out
