import os
import argparse
import json
import torch
import sys

# Adjust path so we can run from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudGAT, FraudCamouflageGNN
from src.models.layers.sage import GraphSAGEModel
from src.utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection GNN")
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to experiment directory containing best_model.pt and config.json")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = os.path.join(args.exp_dir, 'config.json')
    model_path = os.path.join(args.exp_dir, 'best_model.pt')
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Experiment directory must contain config.json and best_model.pt")
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join('data', 'processed', 'graph.pt')
    data = torch.load(data_path).to(device)
    in_channels = data.x.size(1)

    # Instantiate model
    if config['model'] == 'sage':
        model = GraphSAGEModel(in_channels, config.get('hidden_channels', 64)).to(device)
    elif config['model'] == 'gat':
        model = FraudGAT(in_channels, config.get('hidden_channels', 64), heads=config.get('heads', 4), dropout=config.get('dropout', 0.3)).to(device)
    elif config['model'] == 'camouflage':
        model = FraudCamouflageGNN(in_channels, config.get('hidden_channels', 64), heads=config.get('heads', 4), dropout=config.get('dropout', 0.3)).to(device)
    else:
        raise ValueError("Unknown model type")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Evaluating on test set...")
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        test_metrics = compute_metrics(out[data.test_mask], data.y[data.test_mask])

    print("Test Results:")
    print(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  F1-Score:{test_metrics['f1']:.4f}")

    # Save test metrics
    with open(os.path.join(args.exp_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    main()
