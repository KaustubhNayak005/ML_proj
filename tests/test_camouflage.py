import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.layers.camouflage_gat import CamouflageGATConv

def test_camouflage_gat_conv_shape():
    in_channels = 16
    out_channels = 8
    heads = 4
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn((num_nodes, in_channels))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    conv = CamouflageGATConv(in_channels, out_channels, heads=heads, concat=True)
    out = conv(x, edge_index)
    
    assert out.size() == (num_nodes, heads * out_channels)
