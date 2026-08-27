import os
import json
import subprocess
import sys
import torch
from torch_geometric.data import Data
sys.path.append(os.path.abspath('.'))

def test_real_subprocess():
    exp_dir = os.path.join("experiments", "test_wiring")
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({"model": "sage", "hidden_channels": 16}, f)
        
    from src.models.layers.sage import GraphSAGEModel
    model = GraphSAGEModel(10, 16)
    torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
    
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    y = torch.tensor([0, 1, 0, 1, 0])
    test_mask = torch.tensor([False, False, True, True, True])
    data = Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask)
    os.makedirs("data/processed", exist_ok=True)
    torch.save(data, "data/processed/graph.pt")

    eval_cmd = [
        "python", "src/evaluate.py",
        "--exp_dir", exp_dir
    ]
    
    print(f"--- Calling ACTUAL subprocess: {' '.join(eval_cmd)} ---")
    
    # ACTUAL non-mocked subprocess call
    subprocess.run(eval_cmd, check=True)

    # Mimic the reading logic from run_experiments.py
    metrics_path = os.path.join(exp_dir, "test_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            results = json.load(f)
            print(f"\nResults dict populated correctly from real subprocess's test_results.json:")
            print(json.dumps(results, indent=4))
    else:
        print("\ntest_results.json not found!")

if __name__ == '__main__':
    test_real_subprocess()
