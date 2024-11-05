import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN_SIA(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(GIN_SIA, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = [config['hidden_units'][0]] + config['hidden_units']
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        ## additional structural information dimention
        self.info_dim = config['info_dim']
        self.struc_info = config['struc_info']

        print ( "[GIN_SIA.py]######## GIN_SIA " + str(self.struc_info) + " " + str(self.info_dim) + " starts #########" ) 

        train_eps = config['train_eps']
        if config['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features+self.info_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())

                ## add structural information for each layer of GIN
                self.linears.append(Linear(out_emb_dim+self.info_dim, dim_target))

            else :
                input_emb_dim = self.embeddings_dim[layer-1]+self.info_dim
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                ## add structural information for each layer of GIN
                self.linears.append(Linear(out_emb_dim+self.info_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input


    def forward(self, data):
        x, edge_index, batch, f_vector, singular_values, sv_pca, eigen_values, ev_pca = data.x, data.edge_index, data.batch, data.f_vector, data.singular_values, data.sv_pca, data.eigen_values, data.ev_pca

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

        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                ## add structural information for each layer of GIN
                x = self.first_h( torch.cat( [x, repeated_struc_info_torch], dim=1 ) )
                out += F.dropout(self.pooling(self.linears[layer]( torch.cat( [x, repeated_struc_info_torch], dim=1)), batch), p=self.dropout)

            else :
                ## add structural information for each layer of GIN & READOUT layer of GIN
                x = self.convs[layer-1]( torch.cat( [x, repeated_struc_info_torch], dim=1 ), edge_index)
                out += F.dropout(self.linears[layer]( torch.cat( [self.pooling(x, batch), struc_info_torch], dim=1) ), p=self.dropout, training=self.training )

        return out