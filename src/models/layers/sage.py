import torch
import torch.nn as nn

class SAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # GraphSAGE concatenates the node's own features with the aggregated neighbor features,
        # so the linear transformation takes in 2 * in_channels.
        self.lin = nn.Linear(in_channels * 2, out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # 1. Gather neighbor features
        # src_features: [num_edges, in_channels]
        src_features = x[src]
        
        # 2. Aggregate neighbor features (Mean aggregation)
        # Initialize an empty tensor for aggregated features: [num_nodes, in_channels]
        aggr_out = torch.zeros((num_nodes, x.size(1)), device=x.device, dtype=x.dtype)
        
        # We scatter-reduce (mean) the src_features into the dst nodes
        # dst.unsqueeze(1).expand(-1, in_channels) creates the index tensor matching src_features shape
        aggr_out.scatter_reduce_(dim=0, 
                                 index=dst.unsqueeze(1).expand(-1, x.size(1)), 
                                 src=src_features, 
                                 reduce="mean", 
                                 include_self=False)
        
        # 3. Concatenate self features with aggregated neighbor features
        # concat_out: [num_nodes, in_channels * 2]
        concat_out = torch.cat([x, aggr_out], dim=-1)
        
        # 4. Apply linear transformation and non-linearity
        # out: [num_nodes, out_channels]
        out = self.act(self.lin(concat_out))
        
        # Optional: GraphSAGE often L2-normalizes the output embeddings
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        
        return out

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # Binary classification output (logits)
        out = self.classifier(x)
        return out
