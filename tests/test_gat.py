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
