import io
import os
import json
import requests
import zipfile
from pathlib import Path
import networkx as nx
from networkx import normalized_laplacian_matrix

import numpy as np
import torch
from torch.nn import functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.utils import NumpyEncoder
from .data import Data
from .dataloader import DataLoader
from .dataset import GraphDataset, GraphDatasetSubset
from .sampler import RandomSampler
from .tu_utils import parse_tu_data, create_graph_from_tu_data

import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigsh
from sklearn.decomposition import PCA


class GraphDatasetManager:
    def __init__(self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None, seed=42, holdout_test_size=0.1,
                 use_node_degree=False, use_node_attrs=False, use_one=False, precompute_kron_indices=False,
                 max_reductions=10, DATA_DIR='DATA'):

        self.root_dir = Path(DATA_DIR) / self.name
        self.kfold_class = kfold_class
        self.holdout_test_size = holdout_test_size
        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one
        self.precompute_kron_indices = precompute_kron_indices
        self.KRON_REDUCTIONS = max_reductions  # will compute indices for 10 pooling layers --> approximately 1000 nodes

#        print ( "\n[manager.py] Manager is being created " )
        self.outer_k = outer_k
        assert (outer_k is not None and outer_k > 0) or outer_k is None

        self.inner_k = inner_k
        assert (inner_k is not None and inner_k > 0) or inner_k is None

        self.seed = seed

        self.raw_dir = self.root_dir / "raw"
        if not self.raw_dir.exists():
            os.makedirs(self.raw_dir)
            self._download()

#        print ( "[manager.py] raw is done. processed is started------" )

        self.processed_dir = self.root_dir / "processed"
        if not (self.processed_dir / f"{self.name}.pt").exists():
            if not self.processed_dir.exists():
                os.makedirs(self.processed_dir)
            self._process()

#        print ( "[manager.py] processed is done. GraphDataset is started------" )

        self.dataset = GraphDataset(torch.load(
            self.processed_dir / f"{self.name}.pt"))


        splits_filename = self.processed_dir / f"{self.name}_splits.json"
        if not splits_filename.exists():
            self.splits = []
            self._make_splits()
        else:
            self.splits = json.load(open(splits_filename, "r"))


    @property
    def num_graphs(self):
        return len(self.dataset)

    @property
    def dim_target(self):
        if not hasattr(self, "_dim_target") or self._dim_target is None:
            # not very efficient, but it works
            # todo not general enough, we may just remove it
            self._dim_target = np.unique(self.dataset.get_targets()).size
        return self._dim_target

    @property
    def dim_features(self):
        if not hasattr(self, "_dim_features") or self._dim_features is None:
            # not very elegant, but it works
            # todo not general enough, we may just remove it
            self._dim_features = self.dataset.data[0].x.size(1)
        return self._dim_features

    def _process(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError

    def _make_splits(self):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """

        targets = self.dataset.get_targets()
        all_idxs = np.arange(len(targets))

        if self.outer_k is None:  # holdout assessment strategy
            assert self.holdout_test_size is not None

            if self.holdout_test_size == 0:
                train_o_split, test_split = all_idxs, []
            else:
                outer_split = train_test_split(all_idxs,
                                               stratify=targets,
                                               test_size=self.holdout_test_size)
                train_o_split, test_split = outer_split
            split = {"test": all_idxs[test_split], 'model_selection': []}

            train_o_targets = targets[train_o_split]

            if self.inner_k is None:  # holdout model selection strategy
                if self.holdout_test_size == 0:
                    train_i_split, val_i_split = train_o_split, []
                else:
                    train_i_split, val_i_split = train_test_split(train_o_split,
                                                                  stratify=train_o_targets,
                                                                  test_size=self.holdout_test_size)
                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})

            else:  # cross validation model selection strategy
                inner_kfold = self.kfold_class(
                    n_splits=self.inner_k, shuffle=True)
                for train_ik_split, val_ik_split in inner_kfold.split(train_o_split, train_o_targets):
                    split['model_selection'].append(
                        {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

            self.splits.append(split)

        else:  # cross validation assessment strategy

            outer_kfold = self.kfold_class(
                n_splits=self.outer_k, shuffle=True)

            for train_ok_split, test_ok_split in outer_kfold.split(X=all_idxs, y=targets):
                split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                train_ok_targets = targets[train_ok_split]

                if self.inner_k is None:  # holdout model selection strategy
                    assert self.holdout_test_size is not None
                    train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                  stratify=train_ok_targets,
                                                                  test_size=self.holdout_test_size)
                    split['model_selection'].append(
                        {"train": train_i_split, "validation": val_i_split})

                else:  # cross validation model selection strategy
                    inner_kfold = self.kfold_class(
                        n_splits=self.inner_k, shuffle=True)
                    for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split, train_ok_targets):
                        split['model_selection'].append(
                            {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                self.splits.append(split)

        filename = self.processed_dir / f"{self.name}_splits.json"
        with open(filename, "w") as f:
            json.dump(self.splits[:], f, cls=NumpyEncoder)

    def _get_loader(self, dataset, batch_size=1, shuffle=True):
        # dataset = GraphDataset(data)
        sampler = RandomSampler(dataset) if shuffle is True else None

        # 'shuffle' needs to be set to False when instantiating the DataLoader,
        # because pytorch  does not allow to use a custom sampler with shuffle=True.
        # Since our shuffler is a random shuffler, either one wants to do shuffling
        # (in which case he should instantiate the sampler and set shuffle=False in the
        # DataLoader) or he does not (in which case he should set sampler=None
        # and shuffle=False when instantiating the DataLoader)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          shuffle=False,  # if shuffle is not None, must stay false, ow is shuffle is false
                          pin_memory=True)

    def get_test_fold(self, outer_idx, batch_size=1, shuffle=True):
        outer_idx = outer_idx or 0

        idxs = self.splits[outer_idx]["test"]
        test_data = GraphDatasetSubset(self.dataset.get_data(), idxs)

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = self._get_loader(test_data, batch_size, shuffle)

        return test_loader

    def get_model_selection_fold(self, outer_idx, inner_idx=None, batch_size=1, shuffle=True):
        outer_idx = outer_idx or 0
        inner_idx = inner_idx or 0

        idxs = self.splits[outer_idx]["model_selection"][inner_idx]

        train_data = GraphDatasetSubset(self.dataset.get_data(), idxs["train"])
        val_data = GraphDatasetSubset(self.dataset.get_data(), idxs["validation"])

        train_loader = self._get_loader(train_data, batch_size, shuffle)

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = self._get_loader(val_data, batch_size, shuffle)

        return train_loader, val_loader


class TUDatasetManager(GraphDatasetManager):
#    URL = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{name}.zip"
    URL = "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    classification = True

    def _download(self):
        url = self.URL.format(name=self.name)
        response = requests.get(url)
        stream = io.BytesIO(response.content)
        with zipfile.ZipFile(stream) as z:
            for fname in z.namelist():
                z.extract(fname, self.raw_dir)

    def _process(self):
        graphs_data, num_node_labels, num_edge_labels = parse_tu_data(self.name, self.raw_dir)
        targets = graphs_data.pop("graph_labels")

        # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
        max_num_nodes = max([len(v) for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []



        # create whole graph to check number of component
        temp_whole_G = nx.Graph()

        # dictionary to save structural information for each graph
        fiedler_dict = {}
        singular_value_dict = {}    
        eigen_value_dict = {}          

        for i, target in enumerate(targets, 1):
            graph_data = {k: v[i] for (k, v) in graphs_data.items()}

            G = create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels)

            if i % 100 == 0 :
                print ( i," is done..." ) 

            adj = nx.adjacency_matrix ( G )  
            nx_G = nx.Graph( adj )
            num_nodes = nx_G.size()
                
            temp_whole_G = nx.union( temp_whole_G, G )
            num_cc = nx.number_connected_components(nx_G)
            subgraphs_list = [ nx_G.subgraph(c).copy() for c in nx.connected_components(nx_G) ]
 
            f_vector = self.get_fiedler_vector( i, subgraphs_list ) 
            fiedler_dict[i] = f_vector

            singular_value = self.get_singular_value( i, subgraphs_list )      
            singular_value_dict[i] = singular_value          

            eigen_value = self.get_eigen_value( i, subgraphs_list )      
            eigen_value_dict[i] = eigen_value   


            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list


        current_dir = os.getcwd()

        if self.name == 'ENZYMES' :
            fiedler_path = os.path.join(current_dir, 'SI_fv_ENZYMES.json')
            sv_path = os.path.join(current_dir, 'SI_sv_ENZYMES.json')             
            ev_path = os.path.join(current_dir, 'SI_ev_ENZYMES.json')             

        elif self.name == 'PROTEINS_full' : 
            fiedler_path = os.path.join(current_dir, 'SI_fv_PROTEINS.json')
            sv_path = os.path.join(current_dir, 'SI_sv_PROTEINS.json')             
            ev_path = os.path.join(current_dir, 'SI_ev_PROTEINS.json')             

        elif self.name == 'DD' : 
            fiedler_path = os.path.join(current_dir, 'SI_fv_DD.json')
            sv_path = os.path.join(current_dir, 'SI_sv_DD.json')             
            ev_path = os.path.join(current_dir, 'SI_ev_DD.json')             


        with open( fiedler_path, "w") as f:
            json.dump( fiedler_dict, f )
        print ( "[manager.py] Fiedler vector creation is done " )


        with open( sv_path, "w") as f2:   
            json.dump( singular_value_dict, f2 )           
        print ( "[manager.py] Singular value creation is done " ) 


        with open( ev_path, "w") as f3:    
            json.dump( eigen_value_dict, f3 )         
        print ( "[manager.py] Eigen value creation is done " ) 


        ## To perform dimensionality reduction such as PCA, the entire data is required.
        ## However, in the current code, the fiedler vector or singular value is calculated for each graph.
        ## We need to have all the singular values of graph 1, singular values of graph 2, ... to be able to reduce the dimensionality 
        ## in a way that minimizes the variance of these data, but it is not possible in the current code.
        
        ## Therefore, when running "for i, target in enumerate(targets, 1):" for the first time, 
        ## the entire fiedler vector or singular values were obtained and saved as a file, 
        ## and then through second execution of "for i, target in enumerate(targets, 1):", the structure information of the corresponding file are loaded 
        ## and PCA are obtained as shown below.

        current_dir = os.getcwd()

        if self.name == 'ENZYMES' : 
            sv_json_path = os.path.join(current_dir, 'SI_sv_ENZYMES.json')
            with open(sv_json_path, 'r') as sv_file:
                sv_load_dict = json.load( sv_file )  

        elif self.name == 'PROTEINS_full' : 
            sv_json_path = os.path.join(current_dir, 'SI_sv_PROTEINS.json')
            with open(sv_json_path, 'r') as sv_file:
                sv_load_dict = json.load( sv_file )  

        elif self.name == 'DD' : 
            sv_json_path = os.path.join(current_dir, 'SI_sv_DD.json')
            with open(sv_json_path, 'r') as sv_file:
                sv_load_dict = json.load( sv_file )  


        if self.name == 'ENZYMES' : 
            ev_json_path = os.path.join(current_dir, 'SI_ev_ENZYMES.json')
            with open(ev_json_path, 'r') as ev_file:
                ev_load_dict = json.load( ev_file )  

        elif self.name == 'PROTEINS_full' : 
            ev_json_path = os.path.join(current_dir, 'SI_ev_PROTEINS.json')
            with open(ev_json_path, 'r') as ev_file:
                ev_load_dict = json.load( ev_file )  

        elif self.name == 'DD' : 
            ev_json_path = os.path.join(current_dir, 'SI_ev_DD.json')
            with open(ev_json_path, 'r') as ev_file:
                ev_load_dict = json.load( ev_file )  


        print ( "[datasets/manager.py] ########## number of graphs : " , len(sv_load_dict) )

        ## The elements of sv_load_lst are singular values of each graph, and their sizes are all different.
        ## The elements of ev_load_lst are eigen values of each graph, and their sizes are all different.
        ## This is because the number of nodes in each graph is different.

        sv_load_lst = list( sv_load_dict.values() ) 
        ev_load_lst = list( ev_load_dict.values() ) 

        ## Use zero padding to make the size of all graph structure information the same. 
        sv_formatted_lst = []
        ev_formatted_lst = []

        for each_graph in sv_load_lst : 
            if len( each_graph ) <= max_num_nodes : 
                diffzero = max_num_nodes - len( each_graph )
                cntzero = [0]*diffzero
                tmp_lst = each_graph + cntzero
            sv_formatted_lst.append( tmp_lst ) 


        for each_graph in ev_load_lst : 
            if len( each_graph ) <= max_num_nodes : 
                diffzero = max_num_nodes - len( each_graph )
                cntzero = [0]*diffzero
                tmp_lst = each_graph + cntzero
            ev_formatted_lst.append( tmp_lst ) 


        max_cnt = 0
        
        for i in sv_formatted_lst : 
            if len(i) == max_num_nodes : 
                max_cnt = max_cnt + 1

        sv_df = pd.DataFrame( sv_formatted_lst ) 
        ev_df = pd.DataFrame( ev_formatted_lst ) 

        pca_model_sv = PCA( n_components= min( max_num_nodes, len(sv_formatted_lst) ) ) 
        pca_model_ev = PCA( n_components= min( max_num_nodes, len(ev_formatted_lst) ) ) 

        fitted_pca_sv = pca_model_sv.fit_transform( sv_df ) 
        fitted_pca_ev = pca_model_ev.fit_transform( ev_df ) 

        df_pca_sv = pd.DataFrame ( fitted_pca_sv ) 
        df_pca_ev = pd.DataFrame ( fitted_pca_ev )


        temp_whole_G = nx.Graph()
        
        for i, target in enumerate(targets, 1):
            graph_data = {k: v[i] for (k, v) in graphs_data.items()}

            G = create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels)

            if i % 100 == 0 :
                print ( i," is done..." ) 

            adj = nx.adjacency_matrix ( G )  
            nx_G = nx.Graph( adj )
            num_nodes = nx_G.size()

                
            temp_whole_G = nx.union( temp_whole_G, G )
            num_cc = nx.number_connected_components(nx_G)
            subgraphs_list = [ nx_G.subgraph(c).copy() for c in nx.connected_components(nx_G) ]

            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                data = self._to_data(G, df_pca_sv.loc[i-1], df_pca_ev.loc[i-1] )
                dataset.append(data)

        torch.save(dataset, self.processed_dir / f"{self.name}.pt")


    def get_fiedler_vector( self, i, subgraphs_list ) :

        final_f_vector = np.array([])

        for g in subgraphs_list : 
            ## If g is a graph composed of a single node, this node does not appear in the adjacency node matrix full_A.txt. 
            ## In other words, it is a single node subgraph with no connected nodes.
            num_nodes = g.number_of_nodes()
            if num_nodes < 2 : 
                continue

            temp_f_vector = nx.fiedler_vector(g)
            temp_f_vector.sort()

            final_f_vector = np.concatenate( (final_f_vector, temp_f_vector) )

        final_f_vector = list( final_f_vector ) 
        formatted_f_vector = [ '%.6f' % elem for elem in final_f_vector ]
        formatted_f_vector = [ float(elem) for elem in formatted_f_vector ]

        return formatted_f_vector


    def get_singular_value( self, i, subgraphs_list ) :

        final_sv = np.array([])

        for g in subgraphs_list : 
            ## If g is a graph composed of a single node, this node does not appear in the adjacency node matrix full_A.txt. 
            ## In other words, it is a single node subgraph with no connected nodes.
            num_nodes = g.number_of_nodes()
            if num_nodes < 2 : 
                continue
            
            sparse_adj_g = nx.adjacency_matrix( g )
            dense_adj_g = csc_matrix( sparse_adj_g.toarray().astype(np.float64) )

            temp_u, temp_sv, temp_vt = svds( dense_adj_g, k=(num_nodes-1) )
            final_sv = np.concatenate( (final_sv, temp_sv) ) 

        final_sv = list( final_sv ) 
        formatted_sv = [ '%.6f' % elem for elem in final_sv ]
        formatted_sv = [ float(elem) for elem in formatted_sv ]
        formatted_sv.reverse()

        return formatted_sv

    def get_eigen_value( self, i, subgraphs_list ) :

        final_ev = np.array([])

        for g in subgraphs_list : 
            ## If g is a graph composed of a single node, this node does not appear in the adjacency node matrix full_A.txt. 
            ## In other words, it is a single node subgraph with no connected nodes.
            num_nodes = g.number_of_nodes()
            if num_nodes < 2 : 
                continue
            
            sparse_adj_g = nx.adjacency_matrix( g )
            dense_adj_g = csc_matrix( sparse_adj_g.toarray().astype(np.float64) )

            temp_w, temp_v = eigsh( dense_adj_g.toarray(), k=(num_nodes-1) )
            final_ev = np.concatenate( (final_ev, temp_w) ) 

        final_ev = list( final_ev ) 
        formatted_ev = [ '%.6f' % elem for elem in final_ev ]
        formatted_ev = [ float(elem) for elem in formatted_ev ]
        formatted_ev.sort( reverse=True )

        return formatted_ev


    def _to_data(self, G, g_pca_sv, g_pca_ev ):

        datadict = {}

        node_features = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        datadict.update(x=node_features)

        if G.laplacians is not None:
            datadict.update(laplacians=G.laplacians)
            datadict.update(v_plus=G.v_plus)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        if G.has_edge_attrs:
            edge_attr = G.get_edge_attr()
            datadict.update(edge_attr=edge_attr)

        target = G.get_target(classification=self.classification)
        datadict.update(y=target)


        ## Add structural information to datadict
        adj = nx.adjacency_matrix ( G )  
        nx_G = nx.Graph( adj )
        num_cc = nx.number_connected_components(nx_G)
            
        subgraphs_list = [ nx_G.subgraph(c).copy() for c in nx.connected_components(nx_G) ]
            
        f_vector = self.get_fiedler_vector_indataclass( subgraphs_list ) 
        singular_values = self.get_singular_value_indataclass( subgraphs_list )
        eigen_values = self.get_eigen_value_indataclass( subgraphs_list )

        datadict.update( f_vector=f_vector )
        datadict.update( singular_values=singular_values )
        datadict.update( eigen_values=eigen_values )

        datadict.update( sv_pca = g_pca_sv ) 
        datadict.update( ev_pca = g_pca_ev ) 

        data = Data(**datadict)

        return data


    def get_fiedler_vector_indataclass( self, subgraphs_list ) :

        final_f_vector = np.array([])

        for g in subgraphs_list : 
            ## If g is a graph composed of a single node, this node does not appear in the adjacency node matrix full_A.txt. 
            ## In other words, it is a single node subgraph with no connected nodes.
            num_nodes = g.number_of_nodes()
            if num_nodes < 2 : 
                continue

            temp_f_vector = nx.fiedler_vector(g)
            temp_f_vector.sort()
            final_f_vector = np.concatenate( (final_f_vector, temp_f_vector) )

        final_f_vector = list( final_f_vector ) 
        formatted_f_vector = [ '%.6f' % elem for elem in final_f_vector ]
        formatted_f_vector = [ float(elem) for elem in formatted_f_vector ]

        return formatted_f_vector


    def get_singular_value_indataclass( self, subgraphs_list ) :

        final_sv = np.array([])

        for g in subgraphs_list : 
            ## If g is a graph composed of a single node, this node does not appear in the adjacency node matrix full_A.txt. 
            ## In other words, it is a single node subgraph with no connected nodes.
            num_nodes = g.number_of_nodes()
            if num_nodes < 2 : 
                continue
            
            sparse_adj_g = nx.adjacency_matrix( g )
            dense_adj_g = csc_matrix( sparse_adj_g.toarray().astype(np.float64) )

            temp_u, temp_sv, temp_vt = svds( dense_adj_g, k=(num_nodes-1) )
            final_sv = np.concatenate( (final_sv, temp_sv) ) 

        final_sv = list( final_sv ) 
        formatted_sv = [ '%.6f' % elem for elem in final_sv ]
        formatted_sv = [ float(elem) for elem in formatted_sv ]
        formatted_sv.reverse()

        return formatted_sv


    def get_eigen_value_indataclass( self, subgraphs_list ) :

        final_ev = np.array([])

        for g in subgraphs_list : 
            ## If g is a graph composed of a single node, this node does not appear in the adjacency node matrix full_A.txt. 
            ## In other words, it is a single node subgraph with no connected nodes.
            num_nodes = g.number_of_nodes()
            if num_nodes < 2 : 
                continue
            
            sparse_adj_g = nx.adjacency_matrix( g )
            dense_adj_g = csc_matrix( sparse_adj_g.toarray().astype(np.float64) )

            temp_w, temp_v = eigsh( dense_adj_g.toarray(), k=(num_nodes-1) )
            final_ev = np.concatenate( (final_ev, temp_w) ) 

        final_ev = list( final_ev ) 
        formatted_ev = [ '%.6f' % elem for elem in final_ev ]
        formatted_ev = [ float(elem) for elem in formatted_ev ]

        return formatted_ev


    def _precompute_kron_indices(self, G):
        laplacians = []  # laplacian matrices (represented as 1D vectors)
        v_plus_list = []  # reduction matrices

        X = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        lap = torch.Tensor(normalized_laplacian_matrix(G).todense())  # I - D^{-1/2}AD^{-1/2}
        # print(X.shape, lap.shape)

        laplacians.append(lap)

        for _ in range(self.KRON_REDUCTIONS):
            if lap.shape[0] == 1:  # Can't reduce further:
                v_plus, lap = torch.tensor([1]), torch.eye(1)
                # print(lap.shape)
            else:
                v_plus, lap = self._vertex_decimation(lap)
                # print(lap.shape)
                # print(lap)

            laplacians.append(lap.clone())
            v_plus_list.append(v_plus.clone().long())

        return laplacians, v_plus_list

    # For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
    # A Laplacian matrix is symmetric (=> diagonalizable)
    # and dominant eigenvalue (true in most cases? can we enforce it?)
    # => we have sufficient conditions for power method to converge
    def _power_iteration(self, A, num_simulations=30):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = torch.mm(A, b_k)

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def _vertex_decimation(self, L):

        max_eigenvec = self._power_iteration(L)
        v_plus, v_minus = (max_eigenvec >= 0).squeeze(), (max_eigenvec < 0).squeeze()

        # print(v_plus, v_minus)

        # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
        if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
            if torch.sum(v_minus) == 0.:
                assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
                # I assumed v_minus should have ones, but this is not necessarily the case. So I added this if
                return torch.ones(v_minus.shape), L
            else:
                return v_minus, L

        L_plus_plus = L[v_plus][:, v_plus]
        L_plus_minus = L[v_plus][:, v_minus]
        L_minus_minus = L[v_minus][:, v_minus]
        L_minus_plus = L[v_minus][:, v_plus]

        L_new = L_plus_plus - torch.mm(torch.mm(L_plus_minus, torch.inverse(L_minus_minus)), L_minus_plus)

        return v_plus, L_new

    def _precompute_assignments(self):
        pass


class NCI1(TUDatasetManager):
    name = "NCI1"
    _dim_features = 37
    _dim_target = 2
    max_num_nodes = 111


class RedditBinary(TUDatasetManager):
    name = "REDDIT-BINARY"
    _dim_features = 1
    _dim_target = 2
    max_num_nodes = 3782


class Reddit5K(TUDatasetManager):
    name = "REDDIT-MULTI-5K"
    _dim_features = 1
    _dim_target = 5
    max_num_nodes = 3648


class Proteins(TUDatasetManager):
    name = "PROTEINS_full"
    _dim_features = 3
    _dim_target = 2
    max_num_nodes = 620


class DD(TUDatasetManager):
    name = "DD"
    _dim_features = 89
    _dim_target = 2
    max_num_nodes = 5748


class Enzymes(TUDatasetManager):
    name = "ENZYMES"
    _dim_features = 21  # 18 attr + 3 labels
    _dim_target = 6
    max_num_nodes = 126


class IMDBBinary(TUDatasetManager):
    name = "IMDB-BINARY"
    _dim_features = 1
    _dim_target = 2
    max_num_nodes = 136


class IMDBMulti(TUDatasetManager):
    name = "IMDB-MULTI"
    _dim_features = 1
    _dim_target = 3
    max_num_nodes = 89


class Collab(TUDatasetManager):
    name = "COLLAB"
    _dim_features = 1
    _dim_target = 3
    max_num_nodes = 492