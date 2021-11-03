import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        self.num_layers = num_layers

        super(GCN, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.convs = ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-1)] + 
                                   [GCNConv(hidden_dim, output_dim)])
        self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = self.linear(x)
        for layer_idx in range(self.num_layers - 1):
            out = self.convs[layer_idx](out, adj_t)
            out = self.bns[layer_idx](out)
            out = F.relu(out)
            out = F.dropout(out, p = self.dropout)
        out = self.convs[-1](out, adj_t)
        if self.return_embeds:
            return out
        return self.softmax(out)
