from torch_geometric import data

class Data(data.Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 v_outs=None,
                 e_outs=None,
                 g_outs=None,
                 o_outs=None,
                 laplacians=None,
                 v_plus=None,
                 singular_values=None,      #### Added for SICGNN
                 f_vector=None,             #### Added for SICGNN
                 sv_pca=None,               #### Added for SICGNN
                 eigen_values=None,         #### Added for SICGNN
                 ev_pca=None,               #### Added for SICGNN
                  **kwargs):

        additional_fields = {
            'v_outs': v_outs,
            'e_outs': e_outs,
            'g_outs': g_outs,
            'o_outs': o_outs,
            'laplacians': laplacians,
            'v_plus': v_plus

        }

        #### Add structural information for SICGNN
        self.f_vector = f_vector
        self.singular_values = singular_values
        self.sv_pca = sv_pca
        self.eigen_values = eigen_values 
        self.ev_pca = ev_pca  

        super().__init__(x, edge_index, edge_attr, y, **additional_fields)


class Batch(data.Batch):
    @staticmethod
    def from_data_list(data_list, follow_batch=[]):


        laplacians = None
        v_plus = None

        #### Add for SICGNN
        f_vector = None
        singular_values = None
        eigen_values = None
        sv_pca = None
        ev_pca = None


        if 'laplacians' in data_list[0]:
            laplacians = [d.laplacians[:] for d in data_list]
            v_plus = [d.v_plus[:] for d in data_list]

        temp_i = 0
        copy_data = []
        for d in data_list:
            copy_data.append(Data(x=d.x,
                                  y=d.y,
                                  edge_index=d.edge_index,
                                  edge_attr=d.edge_attr,
                                  v_outs=d.v_outs,
                                  g_outs=d.g_outs,
                                  e_outs=d.e_outs,
                                  f_vector=d.f_vector,                  #### Added for SICGNN
                                  singular_values=d.singular_values,    #### Added for SICGNN
                                  sv_pca=d.sv_pca,                      #### Added for SICGNN
                                  eigen_values=d.eigen_values,          #### Added for SICGNN
                                  ev_pca=d.ev_pca,                      #### Added for SICGNN
                                  o_outs=d.o_outs)
                             )
            temp_i = temp_i + 1

        batch = data.Batch.from_data_list(copy_data, follow_batch=follow_batch)
        batch['laplacians'] = laplacians
        batch['v_plus'] = v_plus

        return batch
