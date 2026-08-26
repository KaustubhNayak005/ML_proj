import torch
import torch.nn as nn
import torch.nn.functional as F

class GATConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 concat: bool = True, dropout: float = 0.0, add_self_loops: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear projection for each head
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        
        if concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
            
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        """
        num_nodes = x.size(0)
        
        if self.add_self_loops:
            self_loop_edges = torch.arange(num_nodes, device=x.device, dtype=torch.long)
            self_loop_edges = self_loop_edges.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loop_edges], dim=1)

        # 1. Linear transformation
        # x_proj: [num_nodes, heads, out_channels]
        x_proj = self.lin(x).view(-1, self.heads, self.out_channels)

        # 2. Compute attention scores for src and dst nodes
        # alpha_src/dst: [num_nodes, heads, 1]
        alpha_src = (x_proj * self.att_src).sum(dim=-1, keepdim=True)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1, keepdim=True)

        src, dst = edge_index[0], edge_index[1]

        # Edge attention scores: [num_edges, heads, 1]
        alpha = alpha_src[src] + alpha_dst[dst]
        alpha = self.leaky_relu(alpha)

        # Softmax over neighborhood
        # Using a stable softmax implementation via scatter
        alpha_max = torch.zeros((num_nodes, self.heads, 1), device=x.device, dtype=x.dtype)
        alpha_max.scatter_reduce_(0, dst.view(-1, 1, 1).expand(-1, self.heads, 1), alpha, reduce="amax", include_self=False)
        alpha_exp = torch.exp(alpha - alpha_max[dst])
        
        alpha_sum = torch.zeros((num_nodes, self.heads, 1), device=x.device, dtype=x.dtype)
        alpha_sum.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.heads, 1), alpha_exp)
        
        alpha_softmax = alpha_exp / (alpha_sum[dst] + 1e-16)
        
        self._alpha = alpha_softmax
        alpha_softmax = F.dropout(alpha_softmax, p=self.dropout, training=self.training)

        # 3. Message passing
        # messages: [num_edges, heads, out_channels]
        messages = x_proj[src] * alpha_softmax
        
        # Aggregate
        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.heads, self.out_channels), messages)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        
        return out
