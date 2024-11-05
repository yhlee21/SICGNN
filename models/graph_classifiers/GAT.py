import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool
from torch_geometric.nn import GATConv 

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import json

class GAT(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])

        self.in_head = 1 ## 2 for hyper-parameter

        for i in range(num_layers):
            dim_input = (dim_features) if i == 0 else (dim_embedding)

            print ( "[GAT.py] dim_features : " + str(dim_features ) ) 

            if i == 0 :
                conv = GATConv(dim_input, dim_embedding*self.in_head)

            else :
                conv = GATConv(dim_input*self.in_head, dim_embedding*self.in_head)

            self.layers.append(conv)


        print ( "[GAT.py]######## GAT starts!!  #########" ) 

        self.fc1 = nn.Linear(num_layers * dim_embedding * self.in_head, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)


    def forward(self, data):
        x, edge_index, batch, f_vector, singular_values, sv_pca, eigen_values, ev_pca = data.x, data.edge_index, data.batch, data.f_vector, data.singular_values, data.sv_pca, data.eigen_values, data.ev_pca
        y = data.y

        processed_f_vector = []
        processed_singular_values = []
        processed_sv_pca = []
        processed_eigen_values = []
        processed_ev_pca = []

        cuda_loc = x.get_device()
        cuda_str = 'cuda:' + str(cuda_loc)

        x_all = []

        for i, layer in enumerate(self.layers):

            x = layer(x, edge_index)

            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
