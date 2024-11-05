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

class GAT_SII(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']

        ## additional structural information dimention
        self.info_dim = config['info_dim']
        self.struc_info = config['struc_info']

        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])

        self.in_head = 1 ## hyper-parameter

        for i in range(num_layers):
            dim_input = (dim_features + self.info_dim) if i == 0 else (dim_embedding)

            if i == 0 :
                conv = GATConv(dim_input, dim_embedding*self.in_head)
            else :
                conv = GATConv(dim_input*self.in_head + self.info_dim, dim_embedding*self.in_head)

            self.layers.append(conv)

        print ( "[GAT_SII.py]######## GAT_SII " + str(self.struc_info) + " " + str(self.info_dim) + " starts  #########" ) 

        self.fc1 = nn.Linear(num_layers * (dim_embedding*self.in_head) , dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)


    def forward(self, data):
        x, edge_index, batch, f_vector, singular_values, sv_pca, eigen_values, ev_pca = data.x, data.edge_index, data.batch, data.f_vector, data.singular_values, data.sv_pca, data.eigen_values, data.ev_pca
        y = data.y

        processed_struc_info = []

        if self.struc_info == 'EigenValue' : 
            struc_info_values = eigen_values.copy()

        elif self.struc_info == 'EigenValueSorted' :
            struc_info_values = eigen_values.copy()

        elif self.struc_info == 'EigenValuePCA' : 
            struc_info_values = ev_pca.copy()

        elif self.struc_info == 'SingularValue' : 
            struc_info_values = singular_values.copy()

        elif self.struc_info == 'SingularValueSorted' : 
            struc_info_values = singular_values.copy()

        elif self.struc_info == 'SingularValuePCA' : 
            struc_info_values = sv_pca.copy()

        elif self.struc_info == 'FiedlerValue' : 
            struc_info_values = f_vector.copy()

        elif self.struc_info == 'FiedlerValueSorted' : 
            struc_info_values = f_vector.copy()
        
        else : 
            print ( "Wrong Structure Information!" ) 


        for each_struc_value in struc_info_values : 
           ## for sorted_eigenvalues
            if "Sorted" in self.struc_info :
                each_struc_value = sorted( each_struc_value, reverse=True ) 

            if len( each_struc_value ) >= self.info_dim :
                temp_struc_value = each_struc_value[:self.info_dim]
            else : 
                diff = self.info_dim - len(each_struc_value)
                listofzeros = [0] * diff
                temp_struc_value = each_struc_value + listofzeros

            processed_struc_info.append( temp_struc_value )


        if (self.struc_info == 'EigenValuePCA') or (self.struc_info == 'SingularValuePCA') : 
            struc_info_torch = torch.tensor( processed_struc_info ).float() 
        else : 
            struc_info_torch = torch.tensor( processed_struc_info ) 


        cuda_loc = x.get_device()
        cuda_str = 'cuda:' + str(cuda_loc)

        #### CUDA GPU version ####
        struc_info_torch = struc_info_torch.to(cuda_str)

        repeated_struc_info_torch = torch.index_select( struc_info_torch, 0, batch ) 
        repeated_struc_info_torch = repeated_struc_info_torch.to(cuda_str)

        x_all = []

        for i, layer in enumerate(self.layers):
            ## add structural information for each layer of GNN
            x = layer(  torch.cat( [x, repeated_struc_info_torch], dim=1 ), edge_index)

            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x