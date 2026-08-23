import sys
import os
import torch
import torch.nn as nn

# Add project root to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

from src.models.layers.sage import GraphSAGEModel

def test_sage():
    # Tiny synthetic 5-node graph
    num_nodes = 5
    in_channels = 16
    hidden_channels = 32
    
    # 5 nodes, 16 features
    x = torch.randn((num_nodes, in_channels), requires_grad=True)
    
    # Edges: 
    # 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 0
    # Also some bidirectional edges for good measure
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]
    ], dtype=torch.long)
    
    # Random binary labels for the 5 nodes
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]])
    
    model = GraphSAGEModel(in_channels, hidden_channels)
    
    # Forward pass
    logits = model(x, edge_index)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    assert logits.shape == (num_nodes, 1), f"Expected shape {(num_nodes, 1)}, got {logits.shape}"
    
    # Compute loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are non-zero
    assert x.grad is not None, "Input gradients are None"
    assert torch.sum(torch.abs(x.grad)) > 0, "Input gradients are all zero"
    
    # Check that model weights have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Gradient for {name} is all zero"
        
    print("GraphSAGE synthetic graph test passed successfully!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradients computed successfully.")

if __name__ == "__main__":
    test_sage()
