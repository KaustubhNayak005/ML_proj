import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from datetime import datetime

# Adjust path so we can run from anywhere
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN
from src.models.layers.sage import GraphSAGEModel
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fraud Detection GNN")
    parser.add_argument('--config', type=str, help="Path to config YAML file")
    parser.add_argument('--model', type=str, choices=['sage', 'gat', 'camouflage'], default='sage')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()

def load_config(args):
    config = vars(args)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
    
    if config['run_name'] is None:
        config['run_name'] = f"{config['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return config

def main():
    args = parse_args()
    config = load_config(args)
    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create experiment dir
    exp_dir = os.path.join('experiments', config['run_name'])
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Load data
    data_path = os.path.join('data', 'processed', 'graph.pt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph data not found at {data_path}. Run build_graph.py first.")
    
    data = torch.load(data_path, weights_only=False).to(device)
    print(f"Loaded graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges.")

    in_channels = data.x.size(1)

    # Instantiate model
    if config['model'] == 'sage':
        model = GraphSAGEModel(in_channels, config['hidden_channels']).to(device)
    elif config['model'] == 'gat':
        model = FraudGAT(in_channels, config['hidden_channels'], heads=config['heads'], dropout=config['dropout']).to(device)
    elif config['model'] == 'camouflage':
        model = FraudCamouflageGNN(in_channels, config['hidden_channels'], heads=config['heads'], dropout=config['dropout']).to(device)
    else:
        raise ValueError("Unknown model type")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)

    # Calculate class weights for BCE (handling class imbalance)
    train_labels = data.y[data.train_mask]
    num_pos = train_labels.sum().item()
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Starting training...")
    best_val_pr_auc = 0.0
    metrics_log = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index).squeeze()
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask].float())
            val_metrics = compute_metrics(out[data.val_mask], data.y[data.val_mask])

        log_str = (f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | "
                   f"Val Loss: {val_loss.item():.4f} | "
                   f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                   f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(log_str)

        metrics_log.append({
            'epoch': epoch,
            'train_loss': loss.item(),
            'val_loss': val_loss.item(),
            **val_metrics
        })

        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            print("  --> Saved new best model")

    # Save metrics
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print("Training finished.")

if __name__ == "__main__":
    main()
