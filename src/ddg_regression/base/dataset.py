from pathlib import Path
import struct
import numpy as np
from scipy.spatial.distance import cdist
import torch as th
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import pandas as pd
import sys
from torch_geometric.utils import remove_self_loops
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from collections import defaultdict
import random

# import code needed for typing atoms
if __name__ == '__main__':
    import utils
    from atom_types import Typer
else:
    try:
        from . import utils
        from .atom_types import Typer
    except:
        from base import utils
        from base.atom_types import Typer


class ddgData(Data):
    """ Aggregates wt and mut graphs, returning
            graph-specific edge indices, node features, edge attributes and node-coordinates
            global ddG label and paths to the wt/mut pdb files
    """

    def __init__(self, graph_list: list):
        super(ddgData, self).__init__()

        for i in range(2): # iterate over two graphs (wt & mut)
            y = graph_list[i].y # ddG label
            
            self[f'edge_index_{i}'] = graph_list[i].edge_index # indices of source & destination nodes defining edges
            self[f'x_{i}'] = graph_list[i].x # node features: whether node is on pr1/pr2, one-hot encoded type
            self[f'edge_attr_{i}'] = graph_list[i].edge_attr # whether edge is intra (0) or inter (1) protein
            self[f'pos_{i}'] = graph_list[i].pos # node xyz coordinates
            
        self.y = y # ddG label
        self.pdb_wt = graph_list[0].pdb_file # 0 : wt; path to pdb file
        self.pdb_mut = graph_list[1].pdb_file # 1: mut; path to pdb file
    
    def __inc__(self, key, value):
        if key == 'edge_index_0':
            return self.x_0.size(0)
        if key == 'edge_index_1':
            return self.x_1.size(0)
        
        else:
            return super().__inc__(key, value)


class ddgDataSet(Dataset):

    def __init__(self,
        interaction_dist: float=4, # cutoff interaction distance for generating graphs, default 4 A
        graph_mode: str='int_mut', # graph generation mode
        typing_mode='lmg', # mode for atom typing, default LitMolGrid (lmg); other option is res
        cache_frames: bool=False, 
        rough_search: bool=False,
        force_recalc: bool = False,
        **kwargs):

        self.type_map = utils.get_type_map()
        if typing_mode == 'lmg':
            self.node_feature_size = 12  # 11 atom types and 1/0 for binding partner ID
        elif typing_mode == 'res_type':
            self.node_feature_size = 22  # 20 aa, 1 for non-canonical aa, 1/0 for binding partner ID
        self.mut_defs = [] # store information about entry
        self.labels = [] # ddG labels
        self.edge_dim = 2  # intra- vs inter-protein edges
        self.interaction_dist = interaction_dist
        self.typing_mode = typing_mode # typing mode
        self.graph_mode = graph_mode # graph_generation mode 
        self.cache = {}
        self.cache_frames = cache_frames
        self.speedup = rough_search

        self.graph_generation_function_dict = { # options for graph generation
            'int_mut': self._get_int_mut_site_graph,
        }

        self.aa_map = { # map AAs to numbers
            'ALA': 0,
            'ARG': 1,
            'ASN': 2,
            'ASP': 3,
            'CYS': 4,
            'GLU': 5,
            'GLN': 6,
            'GLY': 7,
            'HIS': 8,
            'ILE': 9,
            'LEU': 10,
            'LYS': 11,
            'MET': 12,
            'PHE': 13,
            'PRO': 14,
            'SER': 15,
            'THR': 16,
            'TRP': 17,
            'TYR': 18,
            'VAL': 19,
            'XXX': 20,
        }

        
    def _get_int_mut_site_graph(self, pr1_df: pd.DataFrame, pr2_df: pd.DataFrame, mut_site_df: pd.DataFrame):
        """Return graph composed of nodes at mutation site and surrounding neighborhood nodes (inter- and intra-protein edges)

        Args:
            pr1_df (pd.DataFrame): Protein 1 dataframe
            pr2_df (pd.DataFrame): Protein 2 dataframe
        """

        # obtain coordinates of pr1, pr2 and the mutation site (on pr1)
        coords_pr1 = pr1_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_pr2 = pr2_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_mut_site = mut_site_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)

        #### ID NODES ####
        # dist between mutation and points on same protein (pr1)
        dist_between_mut_pr1 = cdist(coords_mut_site, coords_pr1)

        # identifying nodes within distance of mut on same protein
        _, intra_pr1 = np.where(dist_between_mut_pr1 < self.interaction_dist) # _ is num of mut_site_df
        pr1_node_idx = sorted(np.unique(intra_pr1))
        pr1_nodes = pr1_df.iloc[pr1_node_idx, :]
        coords_pr1 = coords_pr1[pr1_node_idx, :] # overwrite coords_pr1 with nodes only

        # dist between pr1 nodes (mut site + everything within distance) and points on protein 2
        dist_between_pr1_nodes_pr2 = cdist(coords_pr1, coords_pr2)

        # identifying nodes within distance of pr1 nodes
        _, inter_pr2 = np.where(dist_between_pr1_nodes_pr2 < self.interaction_dist)
        pr2_node_idx = sorted(np.unique(inter_pr2))
        pr2_nodes = pr2_df.iloc[pr2_node_idx, :]
        coords_pr2 = coords_pr2[pr2_node_idx, :] # overwrite coords_pr2 with nodes only

        #### MAKE FEATURE VECTOR ####
        # features: one-hot encoded lmg atome type
        node_features_pr1 = self._get_node_features(pr1_nodes)
        node_features_pr2 = self._get_node_features(pr2_nodes)

        # is the node on pr1 or pr2?
        node_bool_pr1 = np.zeros((len(coords_pr1), 1))
        node_bool_pr2 = np.ones((len(coords_pr2), 1))

        # features: node xyz coordinates [cols 0-2], one-hot encoded lmg types [cols 3-13], label of whether pr1/pr2 [col 14]
        feats_pr1 = np.concatenate([coords_pr1, node_features_pr1, node_bool_pr1], axis=-1)
        feats_pr2 = np.concatenate([coords_pr2, node_features_pr2, node_bool_pr2], axis=-1)

        feats = np.concatenate([feats_pr1, feats_pr2], axis=-2)

        #### MAKE EDGE LIST ####
        
        # 1 inter: interactions between pr1 & pr2
        dist_inter = cdist(coords_pr1, coords_pr2)
        inter_src_pr1, inter_dst_pr2 = np.where(dist_inter < self.interaction_dist)

        # 2 intra_pr1: interactions between pr1 nodes
        dist_intra_pr1 = cdist(coords_pr1, coords_pr1)
        intra_src_pr1, intra_dst_pr1 = np.where(dist_intra_pr1 < self.interaction_dist)

        # 3 intra_pr2: interactions between pr2 nodes
        dist_intra_pr2 = cdist(coords_pr2, coords_pr2)
        intra_src_pr2, intra_dst_pr2 = np.where(dist_intra_pr2 < self.interaction_dist)

        # don't need node lookup as only final distances (for edges) only measured for nodes in final dataset 
        pr2_offset = len(pr1_nodes)

        # edge source and destination nodes
        edge_src = np.concatenate([
            [i for i in inter_src_pr1],
            [i for i in intra_src_pr1],
            [i + pr2_offset for i in intra_src_pr2]
        ])
        edge_dst = np.concatenate([
            [i + pr2_offset for i in inter_dst_pr2],
            [i for i in intra_dst_pr1],
            [i + pr2_offset for i in intra_dst_pr2]
        ])

        # each edge must be bidirectional
        edge_src_full = np.concatenate([edge_src, edge_dst])
        edge_dst_full = np.concatenate([edge_dst, edge_src])
        edge_indices = np.vstack([edge_src_full, edge_dst_full]).astype(np.float64)

        #### MAKE EDGE ATTRIBUTES ####

        # for now just 1/0 for inter/intra
        edge_attr = np.concatenate([
            np.ones(len(inter_dst_pr2)),
            np.zeros(len(intra_dst_pr1)),
            np.zeros(len(intra_dst_pr2)),
        ])
        
        edge_attr_full = np.expand_dims(np.concatenate([edge_attr, edge_attr]), 1)

        return feats, edge_indices, edge_attr_full
    

    def _get_node_features(self, df: pd.DataFrame):
        mode = self.typing_mode

        if mode == 'lmg':
            types = df['lmg_types'] # get lmg_types column (integers) from parquet df generated via typing in parse_pdb_to_parquet
            types = types.apply(lambda x: self.type_map[x]) # map to ? using type_map dictionary generated with utils.py get_type_map function
            types = np.array(types) # convert df column / series to np array
            types = utils.get_one_hot(types, nb_classes=max(self.type_map.values()) + 1) # one-hot encode
            return types
        elif mode == 'res_type':
            types = df['residue_name'] # get residue name
            types = types.apply(lambda x: self.aa_map[x] if x in self.aa_map.keys() else 20).astype(np.int64) # map to integer as defined in aa_map
            types = np.array(types)
            types = utils.get_one_hot(types, nb_classes=max(self.aa_map.values()) + 1) # one-hot encode
            return types
        else:
            raise NotImplementedError(mode)

            
    def _generate_graph_dict(self, pr1_df: pd.DataFrame, pr2_df: pd.DataFrame, mut_site_df: pd.DataFrame):
        """Generate dictionary of nodes, edge indices and edge attributes in graph

       Args:
            pr1_df (pd.DataFrame): Protein 1 dataframe
            pr2_df (pd.DataFrame): Protein 2 dataframe
        """
        graph_dict = {}
        nodes, edge_ind, edge_attr = self.graph_generation_function_dict[self.graph_mode](pr1_df, pr2_df, mut_site_df)
        ## nodes = nodes with features: node xyz coordinates [cols 0-2], one-hot encoded types [cols 3-13], bool pr1/2 [col 14] 
        ## edge_ind: indices of source & destination nodes defining edges
        ## edge_attr: label of whether intra- or inter-protein edge
        
        graph_dict['nodes'] = nodes
        graph_dict['edge_ind'] = edge_ind
        graph_dict['edge_attr'] = edge_attr

        return graph_dict

    
    def _parse_graph(self, graph_dict: dict, label:np.ndarray, mut_def:dict):
        """Generate parsed graph object in correct format. This is intended to be overwritten 
            by child classes depending on requirements of the downstream models. 
            Base implementation parses the data into a pytorch geometric data instance.

        Args:
            graph_dict: output of _generate_graph_dict, contains nodes (coordinates & features), edge indices and edge attributes of graph
        """

        # remove self loops
        edge_index, edge_attr = remove_self_loops(
                edge_index=th.from_numpy(graph_dict['edge_ind']).long(),
                edge_attr=th.from_numpy(graph_dict['edge_attr'])
            )   
        
        graph = Data( # add information to torch_geometric.data.Data ("A data object describing a homogeneous graph")
            x=th.from_numpy(graph_dict['nodes'][:,3:]), # node features: whether node is on pr1/pr2, one-hot encoded type
            edge_index=edge_index, # source & destination nodes defining edges
            edge_attr=edge_attr, # whether edge is intra (0) or inter (1) protein
            pos=th.from_numpy(graph_dict['nodes'][:,:3]), # node xyz coordinates
            y=th.tensor(label), # ddG label
            wt_mut=mut_def['wt_mut'], # whether wt or mut graph
            pdb_file = str(mut_def['pdb_file']) # path to pdb file (wt or mut)
        )
        
        return graph
    
        
    def __len__(self):
        """ Get number of entries in dataset """
        return len(self.labels)


    def populate(self, input_file: Path, overwrite: bool=False):
        """Extract information from input files and save in mut_defs list of lists

        Args:
            input_file: cvs containing columns: labels, pdb_wt [path to file], pdb_mut [path to file], chain_prot1 [chains in protein 1 (pr1)], chain_prot2 [chains in protein 2 (pr2)]
        """
        
        inf = pd.read_csv(input_file)
        
        labels = inf["labels"].to_list()

        # extract information from input file -> mut_defs list of lists
        ## contains wt/mut label, path to pdb file, chains in pr1/pr2
        mut_defs = []
        for ind, row in inf.iterrows():
            wt_def = ['wt', row['pdb_wt'], row['chain_prot1'], row['chain_prot2'], row['complex']]
            mut_def = ['mut', row['pdb_mut'], row['chain_prot1'], row['chain_prot2'], row['complex']]
            dict_keys = ['wt_mut', 'pdb_file', 'ch_pr1', 'ch_pr2', 'complex']
            mut_defs.append([dict(zip(dict_keys, wt_def)),dict(zip(dict_keys, mut_def))])

        # either overwrite or add to self.mut_defs and self.labels
        if overwrite:
            self.mut_defs = mut_defs
            self.labels = labels
        else:
            if len(self.mut_defs) > 0:
                self.mut_defs += mut_defs
            else:
                self.mut_defs = mut_defs
            if len(self.labels) != 0:
                self.labels = self.labels + labels
            else:
                self.labels = labels

        
    def __aggregate_graphs__(self, graph_list: list):
        """Aggregate wt and mut graphs using ddgData class, returning
            graph-specific edge indices, node features, edge attributes and node-coordinates
            global ddG label and paths to the wt/mut pdb files
        
        Args:
            graph_list: list of wt and mut graphs
        """
        aggregated = ddgData(graph_list)
        return aggregated
    

    def __getitem__(self, idx: int, force_recalc: bool = False):
        """ Generate aggregated graph (wt + mut) for item in dataset
        
        Args:
            idx: index
        """
        
        ##### read in pdbs #####

        # obtain label and mutation info (whether wt/mut, path to pdb file, chains in pr1 & pr2)
        label = self.labels[idx]
        mut_info = self.mut_defs[idx]

        # generate path to typed parquet file
        typed_wt = Path(str(mut_info[0]['pdb_file']).replace('.pdb', '.parquet')) # 0: wt
        typed_mut = Path(str(mut_info[1]['pdb_file']).replace('.pdb', '.parquet')) # 1: mut
        
        ## wt
        # check if typed file in cache
        if self.cache_frames and str(typed_wt) in self.cache and not force_recalc:
            wt_df = self.cache[str(typed_wt)].copy()

        # check if typed file exists
        elif typed_wt.exists() and not force_recalc:
            wt_df = pd.read_parquet(typed_wt)
                            
        # if not create and save typed files
        else:
            wt_df = utils.parse_pdb_to_parquet(mut_info[0]['pdb_file'], typed_wt, lmg_typed=True, ca=False)

        ## mut
        # check if typed file in cache
        if self.cache_frames and str(typed_mut) in self.cache and not force_recalc:
            mut_df = self.cache[str(typed_mut)].copy()

        # check if typed file exists
        elif typed_mut.exists() and not force_recalc:
            mut_df = pd.read_parquet(typed_mut)
                
        # if not create and save typed files
        else:
            mut_df = utils.parse_pdb_to_parquet(mut_info[1]['pdb_file'], typed_mut, lmg_typed=True, ca=False)

        if self.cache_frames:
            if not str(typed_wt) in self.cache:
                self.cache[str(typed_wt)] = wt_df.copy()
            if not str(typed_mut) in self.cache:
                self.cache[str(typed_mut)] = mut_df.copy()
        
        ##### generate graphs #####

        graphs = []
        for mut_def in mut_info:
            
            # get xyz coordinates of pdb file (wt or mut)
            if mut_def['wt_mut'] == 'wt':
                pdb_df = wt_df.copy()
            elif mut_def['wt_mut'] == 'mut':
                pdb_df = mut_df.copy()
            
            # split pdb file into pr1/pr2
            
            ## identify chains in pr1/pr2
            chs_pr1 = []
            for ch in mut_def['ch_pr1']:
                chs_pr1.append(ch)

            chs_pr2 = []
            for ch in mut_def['ch_pr2']:
                chs_pr2.append(ch)
            
            ## set pr1 to be the mutated chain
            mut_chain = mut_def['complex'].split('_')[-1][1]
            
            if mut_chain in chs_pr1:
                pr1_df = pdb_df[pdb_df['chain_id'].isin(chs_pr1)]
                pr2_df = pdb_df[pdb_df['chain_id'].isin(chs_pr2)]
            elif mut_chain in chs_pr2:
                pr1_df = pdb_df[pdb_df['chain_id'].isin(chs_pr2)]
                pr2_df = pdb_df[pdb_df['chain_id'].isin(chs_pr1)]
            else:
                print('Error! Mut chain not found')
                break
            
            ## mut_site_df: df of atoms at mutated site
            mut_ch = mut_def['complex'].split('_')[-1].split('.pdb')[0][1]
            #mut_ch = mut_def['pdb_file'].split('/')[-1].split('_')[-1].split('.pdb')[0][1]
            #mut_ch = mut_def['pdb_file'].split('/')[-1].split('_')[-4] # chain of mut res
            mut_resi = int(mut_def['complex'].split('_')[-1].split('.pdb')[0][2:-1])
            #mut_resi = int(mut_def['pdb_file'].split('/')[-1].split('_')[-2]) # res number of mut res
            
            mut_site_df = pr1_df[(pr1_df['chain_id'] == mut_ch) & (pr1_df['residue_number'] == mut_resi)]
                

            ##### generate graph #####
            graph_dict = self._generate_graph_dict(pr1_df, pr2_df, mut_site_df)
            graphs.append(self._parse_graph(graph_dict, label, mut_def=mut_def))
            
        graph = self.__aggregate_graphs__(graphs)
        return graph
