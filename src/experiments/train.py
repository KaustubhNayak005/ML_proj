import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN
from src.models.layers.sage import GraphSAGEModel
from src.utils.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/processed/graph.pt')
    parser.add_argument('--model', type=str, choices=['sage', 'gat', 'camouflage'], default='camouflage')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading graph from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found. Run build_graph.py first.")
        return
        
    data = torch.load(args.data_path)
    data = data.to(device)
    
    in_channels = data.x.size(1)
    
    # Initialize model
    if args.model == 'sage':
        model = GraphSAGEModel(in_channels, args.hidden_channels).to(device)
    elif args.model == 'gat':
        model = FraudGAT(in_channels, args.hidden_channels, args.heads, args.dropout).to(device)
    elif args.model == 'camouflage':
        model = FraudCamouflageGNN(in_channels, args.hidden_channels, args.heads, args.dropout).to(device)
        
    # Calculate pos_weight for BCEWithLogitsLoss due to class imbalance
    num_pos = int(data.y[data.train_mask].sum())
    num_neg = int((data.y[data.train_mask] == 0).sum())
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).to(device) if num_pos > 0 else torch.tensor([1.0]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_pr_auc = 0.0
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index).squeeze()
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index).squeeze()
            
            train_metrics = compute_metrics(logits[data.train_mask], data.y[data.train_mask])
            val_metrics = compute_metrics(logits[data.val_mask], data.y[data.val_mask])
            val_loss = criterion(logits[data.val_mask], data.y[data.val_mask].float()).item()
            
            if val_metrics['pr_auc'] > best_val_pr_auc:
                best_val_pr_auc = val_metrics['pr_auc']
                # Save best model
                os.makedirs('experiments/checkpoints', exist_ok=True)
                torch.save(model.state_dict(), f'experiments/checkpoints/best_{args.model}.pt')
                
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                  f"Val PR-AUC: {val_metrics['pr_auc']:.4f}")

    print(f"Training completed. Best Validation PR-AUC: {best_val_pr_auc:.4f}")
    
    # Test
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(f'experiments/checkpoints/best_{args.model}.pt'))
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index).squeeze()
        test_metrics = compute_metrics(logits[data.test_mask], data.y[data.test_mask])
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")

if __name__ == '__main__':
    main()
