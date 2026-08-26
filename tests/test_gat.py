import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.layers.gat import GATConv

def test_gat_conv_shape():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn((num_nodes, in_channels))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    conv = GATConv(in_channels, out_channels, heads=heads, concat=True)
    out = conv(x, edge_index)
    
    assert out.size() == (num_nodes, heads * out_channels)
    
    conv_no_concat = GATConv(in_channels, out_channels, heads=heads, concat=False)
    out_no_concat = conv_no_concat(x, edge_index)
    
    assert out_no_concat.size() == (num_nodes, out_channels)


def test_gat_attention_sums_to_1():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 30
    
    x = torch.randn((num_nodes, in_channels))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # dropout=0 to ensure attention sums exactly to 1
    conv = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=0.0)
    out = conv(x, edge_index)
    
    # Calculate sum of attention weights for each destination node
    alpha = conv._alpha # [num_edges + num_nodes(self loops), heads, 1]
    
    # Get the actual edge index used (including self loops)
    dst = torch.cat([edge_index[1], torch.arange(num_nodes, dtype=torch.long)])
    
    # Sum alphas by destination node
    alpha_sums = torch.zeros((num_nodes, heads, 1))
    alpha_sums.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, heads, 1), alpha)
    
    # Assert all sums are close to 1.0 (with a small epsilon for float precision)
    assert torch.allclose(alpha_sums, torch.ones_like(alpha_sums), atol=1e-5), "Attention weights do not sum to 1"

def test_gat_gradient_flow():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn((num_nodes, in_channels), requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    conv = GATConv(in_channels, out_channels, heads=heads, concat=True)
    out = conv(x, edge_index)
    
    loss = out.sum()
    loss.backward()
    
    # Ensure gradients flow to inputs and parameters
    assert x.grad is not None
    assert torch.sum(torch.abs(x.grad)) > 0
    assert conv.lin.weight.grad is not None
    assert conv.att_src.grad is not None
    assert conv.att_dst.grad is not None
